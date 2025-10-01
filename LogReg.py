import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import string

def preprocess_text(text):
    """Preprocess the text data"""
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_prepare_data(train_file, test_file):
    """Load and prepare training and test data"""
    # Load training data
    train_df = pd.read_csv(train_file)
    
    # Load test data
    test_df = pd.read_csv(test_file)
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    return train_df, test_df

def train_aspect_sentiment_model(train_df):
    """Train separate models for aspect classification and sentiment analysis"""
    
    # Preprocess text
    train_df['processed_text'] = train_df['text'].apply(preprocess_text)
    
    # Initialize vectorizer
    vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2))
    
    # Prepare features
    X = vectorizer.fit_transform(train_df['processed_text'])
    
    # Encode aspect categories
    aspect_encoder = LabelEncoder()
    y_aspect = aspect_encoder.fit_transform(train_df['aspectCategory'])
    
    # Encode sentiment
    sentiment_encoder = LabelEncoder()
    y_sentiment = sentiment_encoder.fit_transform(train_df['polarity'])
    
    # Train aspect classifier
    print("Training aspect classifier...")
    aspect_model = LogisticRegression(random_state=42, max_iter=1000)
    aspect_model.fit(X, y_aspect)
    
    # Train sentiment classifier
    print("Training sentiment classifier...")
    sentiment_model = LogisticRegression(random_state=42, max_iter=1000)
    sentiment_model.fit(X, y_sentiment)
    
    return {
        'vectorizer': vectorizer,
        'aspect_model': aspect_model,
        'sentiment_model': sentiment_model,
        'aspect_encoder': aspect_encoder,
        'sentiment_encoder': sentiment_encoder
    }

def predict_on_test(test_df, model_dict):
    """Make predictions on test data"""
    
    # Preprocess test text
    test_df['processed_text'] = test_df['text'].apply(preprocess_text)
    
    # Prepare features
    X_test = model_dict['vectorizer'].transform(test_df['processed_text'])
    
    # Make predictions
    aspect_predictions = model_dict['aspect_model'].predict(X_test)
    sentiment_predictions = model_dict['sentiment_model'].predict(X_test)
    
    # Decode predictions
    aspect_categories = model_dict['aspect_encoder'].inverse_transform(aspect_predictions)
    sentiment_labels = model_dict['sentiment_encoder'].inverse_transform(sentiment_predictions)
    
    return aspect_categories, sentiment_labels

def create_submission_file(test_df, aspect_predictions, sentiment_predictions, output_file):
    """Create submission file in the required format"""
    
    submission_df = test_df.copy()
    submission_df['aspectCategory'] = aspect_predictions
    submission_df['polarity'] = sentiment_predictions
    
    # Keep only the required columns
    submission_df = submission_df[['id', 'text', 'aspectCategory', 'polarity']]
    
    # Save to CSV
    submission_df.to_csv(output_file, index=False)
    print(f"Submission file saved as: {output_file}")
    
    return submission_df

def evaluate_on_dev_set(train_df, test_size=0.2):
    """Evaluate model performance on development set"""
    
    # Preprocess text
    train_df['processed_text'] = train_df['text'].apply(preprocess_text)
    
    # Split data
    X_train, X_dev, y_train_aspect, y_dev_aspect, y_train_sentiment, y_dev_sentiment = train_test_split(
        train_df['processed_text'], 
        train_df['aspectCategory'], 
        train_df['polarity'],
        test_size=test_size, 
        random_state=42,
        stratify=train_df['aspectCategory']
    )
    
    # Initialize vectorizer
    vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_dev_vec = vectorizer.transform(X_dev)
    
    # Encode labels
    aspect_encoder = LabelEncoder()
    sentiment_encoder = LabelEncoder()
    
    y_train_aspect_enc = aspect_encoder.fit_transform(y_train_aspect)
    y_dev_aspect_enc = aspect_encoder.transform(y_dev_aspect)
    
    y_train_sentiment_enc = sentiment_encoder.fit_transform(y_train_sentiment)
    y_dev_sentiment_enc = sentiment_encoder.transform(y_dev_sentiment)
    
    # Train models
    aspect_model = LogisticRegression(random_state=42, max_iter=1000)
    sentiment_model = LogisticRegression(random_state=42, max_iter=1000)
    
    aspect_model.fit(X_train_vec, y_train_aspect_enc)
    sentiment_model.fit(X_train_vec, y_train_sentiment_enc)
    
    # Make predictions
    aspect_pred = aspect_model.predict(X_dev_vec)
    sentiment_pred = sentiment_model.predict(X_dev_vec)
    
    # Decode predictions
    aspect_pred_decoded = aspect_encoder.inverse_transform(aspect_pred)
    sentiment_pred_decoded = sentiment_encoder.inverse_transform(sentiment_pred)
    
    # Calculate accuracy
    aspect_accuracy = (aspect_pred_decoded == y_dev_aspect.values).mean()
    sentiment_accuracy = (sentiment_pred_decoded == y_dev_sentiment.values).mean()
    
    # Calculate overall accuracy (both aspect and sentiment correct)
    overall_accuracy = ((aspect_pred_decoded == y_dev_aspect.values) & 
                       (sentiment_pred_decoded == y_dev_sentiment.values)).mean()
    
    print("\n=== Development Set Evaluation ===")
    print(f"Aspect Classification Accuracy: {aspect_accuracy:.4f}")
    print(f"Sentiment Classification Accuracy: {sentiment_accuracy:.4f}")
    print(f"Overall Accuracy (Both Correct): {overall_accuracy:.4f}")
    
    # Print detailed classification reports
    print("\n=== Aspect Classification Report ===")
    print(classification_report(y_dev_aspect, aspect_pred_decoded))
    
    print("\n=== Sentiment Classification Report ===")
    print(classification_report(y_dev_sentiment, sentiment_pred_decoded))
    
    return overall_accuracy

def main():
    """Main function to run the ABSA system"""
    
    # File paths
    train_file = "contest1_train.csv"
    test_file = "contest1_test.csv"
    output_file = "test_pred.csv"
    
    try:
        # Load data
        print("Loading data...")
        train_df, test_df = load_and_prepare_data(train_file, test_file)
        
        # Evaluate on development set
        print("\nEvaluating model on development set...")
        dev_accuracy = evaluate_on_dev_set(train_df)
        
        # Train final model on all training data
        print("\nTraining final model on all training data...")
        model_dict = train_aspect_sentiment_model(train_df)
        
        # Make predictions on test set
        print("Making predictions on test set...")
        aspect_predictions, sentiment_predictions = predict_on_test(test_df, model_dict)
        
        # Create submission file
        print("Creating submission file...")
        submission_df = create_submission_file(test_df, aspect_predictions, sentiment_predictions, output_file)
        
        # Print some statistics
        print("\n=== Prediction Statistics ===")
        print(f"Total predictions: {len(submission_df)}")
        print(f"Aspect distribution:")
        print(submission_df['aspectCategory'].value_counts())
        print(f"\nSentiment distribution:")
        print(submission_df['polarity'].value_counts())
        
        print(f"\nSubmission file '{output_file}' created successfully!")
        print("Use the following command to evaluate:")
        print(f"python evaluate.py {train_file} {output_file}")
        print(f"python check_id.py {test_file} {output_file}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the CSV files are in the current directory")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()