# absa_groq_pipeline.py
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import ast

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
# -------------------------
FEW_SHOT_PROMPT = """
You are an AI assistant trained to extract aspects and their sentiment from product reviews.
Your task is to identify all aspects mentioned in the review and assign one of the following polarity labels: positive, negative, neutral, conflict.

Important instructions:
1. Only extract aspects that exist in the review.
2. Output each aspect and its polarity on a separate line using strict JSON format: 
   {{"aspectCategory": "<aspect>", "polarity": "<polarity>"}}
3. Do not include any extra text or explanation.
4. Common aspect categories include: food, service, price, ambiance, anecdotes/miscellaneous, but others may appear.
5. If multiple aspects exist, list each one separately.
6. Use lowercase for aspect names and polarity.

Examples:

Review: "The food was delicious but the service was slow and unfriendly."
Output: {{"aspectCategory": "food", "polarity": "positive"}}
Output: {{"aspectCategory": "service", "polarity": "negative"}}

Review: "I loved the ambiance but the food was too salty and expensive."
Output: {{"aspectCategory": "ambiance", "polarity": "positive"}}
Output: {{"aspectCategory": "food", "polarity": "negative"}}
Output: {{"aspectCategory": "price", "polarity": "negative"}}

Now classify this review:
Review: "{text}"
"""

# -------------------------
# Function to classify a single review
# -------------------------
def classify_review(review_text):
    prompt = FEW_SHOT_PROMPT.format(text=review_text)
    response = client.responses.create(
        model="openai/gpt-oss-20b",
        input=prompt
    )
    output_text = response.output_text
    try:
        # Replace "}\nOutput: {" with "},{" in the string
        cleaned_text = output_text.replace("}\nOutput: {", "},{")
        # Wrap with brackets
        cleaned_text = f"[{cleaned_text}]"
        predictions = ast.literal_eval(cleaned_text)
    except Exception as e:
        print(f"Error parsing output: {output_text}, error: {e}")
        predictions = []

# -------------------------
# Load datasets
# -------------------------
df_train_full = pd.read_csv("contest1_train.csv")
df_test       = pd.read_csv("contest1_test.csv")

# Split training set into train/validation
df_train, df_val = train_test_split(df_train_full, test_size=0.2, random_state=42, shuffle=True)

# Save validation ground truth
df_val.to_csv("fewshot_val_ground_truth.csv", index=False)

# -------------------------
# Generate validation predictions
# -------------------------
val_rows = []
for idx, row in tqdm(df_val.iterrows(), total=len(df_val), desc="Validation"):
    review_text = row['text']
    predictions = classify_review(review_text)
    if not predictions:
        predictions = [{"aspectCategory": "food", "polarity": "positive"}]  # fallback
    for pred in predictions:
        val_rows.append({
            "id": row['id'],
            "aspectCategory": pred['aspectCategory'],
            "polarity": pred['polarity']
        })

val_pred = pd.DataFrame(val_rows)
val_pred.to_csv("fewshot_val_pred.csv", index=False)
print("Saved val_pred.csv and val_ground_truth.csv")

# -------------------------
# Generate test predictions
# -------------------------
test_rows = []
for idx, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Test"):
    review_text = row['text']
    predictions = classify_review(review_text)
    if not predictions:
        predictions = [{"aspectCategory": "food", "polarity": "positive"}]  # fallback
    for pred in predictions:
        test_rows.append({
            "id": row['id'],
            "aspectCategory": pred['aspectCategory'],
            "polarity": pred['polarity']
        })

test_pred = pd.DataFrame(test_rows)
test_pred.to_csv("fewshot_test_pred.csv", index=False)
print("Saved test_pred.csv")
