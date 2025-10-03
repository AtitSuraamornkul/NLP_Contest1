import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import time
from openai import RateLimitError, APIError

# -------------------------
# Load API key from .env
# -------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY is None:
    raise ValueError("Please set GROQ_API_KEY in your .env file")

# -------------------------
# Initialize Groq client
# -------------------------
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

# -------------------------
# Enhanced few-shot prompt template
# Note: {{{{ becomes {{ in the final string due to .format() escaping
# -------------------------
FEW_SHOT_PROMPT = """You are an AI assistant trained to extract aspects and their sentiment from product reviews.

Rules:
1. Output ONLY valid JSON, one per line
2. Format: {{"aspectCategory": "<aspect>", "polarity": "<polarity>"}}
3. Use lowercase for aspect names and polarity
4. No explanations or extra text
5. For each sentence, watch for duplicated outputs and aspects
5. Each text can have multiple aspects, identify all possible aspect for each sentence. For the chosen aspect, identify the polarity based on the aspect chosen from the sentence.
- Use only these labels:
    aspectCategory: food, service, price, ambience, anecdotes/miscellaneous
    polarity: positive, negative, neutral, conflict

Examples:

Review: "The food was delicious but the service was slow."
{{"aspectCategory": "food", "polarity": "positive"}}
{{"aspectCategory": "service", "polarity": "negative"}}

Review: "Great ambiance, but overpriced and mediocre food."
{{"aspectCategory": "ambience", "polarity": "positive"}}
{{"aspectCategory": "price", "polarity": "negative"}}
{{"aspectCategory": "food", "polarity": "negative"}}

Now classify:
Review: "{text}"
"""

# -------------------------
# Valid values for validation
# -------------------------
VALID_POLARITIES = {'positive', 'negative', 'neutral', 'conflict'}

# -------------------------
# Function to validate prediction
# -------------------------
def validate_prediction(pred):
    """Validate prediction format and values"""
    if not isinstance(pred, dict):
        return False
    if 'aspectCategory' not in pred or 'polarity' not in pred:
        return False
    
    polarity = pred['polarity'].lower()
    if polarity not in VALID_POLARITIES:
        return False
    
    # Normalize to lowercase
    pred['aspectCategory'] = pred['aspectCategory'].lower()
    pred['polarity'] = polarity
    
    return True

# -------------------------
# Function to classify a single review with retry logic
# -------------------------
def classify_review(review_text, max_retries=3):
    """Classify review with exponential backoff retry logic"""
    prompt = FEW_SHOT_PROMPT.format(text=review_text)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",  # Use valid Groq model
                messages=[
                    {"role": "system", "content": "You are an aspect-based sentiment analysis assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            output_text = response.choices[0].message.content
            
            predictions = []
            for line in output_text.strip().split("\n"):
                line = line.strip()
                if not line or not line.startswith('{'):
                    continue
                try:
                    pred = json.loads(line)
                    if validate_prediction(pred):
                        predictions.append(pred)
                except json.JSONDecodeError:
                    continue
            
            return predictions
            
        except RateLimitError:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Failed after {max_retries} attempts due to rate limit")
                return []
        except APIError as e:
            print(f"API error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []
    
    return []

# -------------------------
# Load datasets
# -------------------------
print("Loading datasets...")
df_train_full = pd.read_csv("contest1_train.csv")
df_test = pd.read_csv("contest1_test.csv")

# Split training set into train/validation
df_train, df_val = train_test_split(df_train_full, test_size=0.2, random_state=42, shuffle=True)

# Save validation ground truth
df_val.to_csv("fewshot_val_ground_truth.csv", index=False)
print(f"Saved fewshot_val_ground_truth.csv ({len(df_val)} samples)")

# -------------------------
# Test on 5 samples first
# -------------------------
print("\n--- Testing on first 5 validation samples ---")
for idx, row in df_val.head(5).iterrows():
    review_text = row['text']
    predictions = classify_review(review_text)
    print(f"\nID: {row['id']}")
    print(f"Review: {review_text[:100]}...")
    if predictions:
        for pred in predictions:
            print(f"  → Aspect: {pred['aspectCategory']}, Polarity: {pred['polarity']}")
    else:
        print("  → No predictions returned.")

# -------------------------
# Generate validation predictions
# -------------------------
print("\n--- Generating validation predictions ---")
val_rows = []
empty_prediction_ids = []

for idx, row in tqdm(df_val.iterrows(), total=len(df_val), desc="Validation"):
    review_text = row['text']
    predictions = classify_review(review_text)
    
    if predictions:
        for pred in predictions:
            val_rows.append({
                "id": row['id'],
                "aspectCategory": pred['aspectCategory'],
                "polarity": pred['polarity']
            })
    else:
        empty_prediction_ids.append(row['id'])

val_pred = pd.DataFrame(val_rows)
val_pred.to_csv("fewshot_val_pred.csv", index=False)
print(f"Saved fewshot_val_pred.csv ({len(val_pred)} predictions)")

if empty_prediction_ids:
    print(f"Warning: {len(empty_prediction_ids)} reviews had no predictions")
    print(f"IDs: {empty_prediction_ids[:10]}...")

# -------------------------
# Generate test predictions
# -------------------------
print("\n--- Generating test predictions ---")
test_rows = []
empty_test_ids = []

for idx, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Test"):
    review_text = row['text']
    predictions = classify_review(review_text)
    
    if predictions:
        for pred in predictions:
            test_rows.append({
                "id": row['id'],
                "aspectCategory": pred['aspectCategory'],
                "polarity": pred['polarity']
            })
    else:
        empty_test_ids.append(row['id'])

test_pred = pd.DataFrame(test_rows)
test_pred.to_csv("test_pred.csv", index=False)
print(f"Saved test_pred.csv ({len(test_pred)} predictions)")

if empty_test_ids:
    print(f"Warning: {len(empty_test_ids)} reviews had no predictions")
    print(f"IDs: {empty_test_ids[:10]}...")