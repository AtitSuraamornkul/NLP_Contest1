#!/usr/bin/env python3
"""
train_logreg_absa.py

Trains two logistic regression classifiers for ABSA:
 - aspectCategory classifier (multiclass)
 - polarity classifier (multiclass: positive/negative/neutral)

Outputs:
 - saved models (joblib): aspect_model.joblib, sentiment_model.joblib
 - test_pred.csv with columns: id,aspectCategory,polarity
 - prints classification reports on a held-out dev set
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
import joblib
import warnings
warnings.filterwarnings("ignore")

def load_data(path):
    df = pd.read_csv(path)
    # Expect columns: id, text, aspectCategory, polarity
    required = {'id', 'text', 'aspectCategory', 'polarity'}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {required}. Found: {df.columns.tolist()}")
    return df

def build_text_pipeline():
    # Tfidf on unigrams + bigrams, max_features optional
    vect = TfidfVectorizer(lowercase=True,
                           stop_words='english',
                           ngram_range=(1,2),
                           max_df=0.95,
                           min_df=2)
    # logistic regression with sag or liblinear depending on solver & size
    clf = LogisticRegression(max_iter=2000, solver='saga', random_state=42, n_jobs=-1)
    pipe = Pipeline([
        ('tfidf', vect),
        ('clf', clf)
    ])
    return pipe

def grid_search_train(X_train, y_train):
    """
    Quick grid search for C parameter; keep small grid so it runs fast.
    """
    pipe = build_text_pipeline()
    param_grid = {
        'clf__C': [0.01, 0.1, 1.0, 5.0],
        'clf__class_weight': [None, 'balanced']
    }
    gs = GridSearchCV(pipe, param_grid, cv=3, scoring='f1_macro', n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)
    print("Best params:", gs.best_params_)
    return gs.best_estimator_

def train_and_save(train_csv, test_csv=None, output_dir='models', use_grid_search=True):
    os.makedirs(output_dir, exist_ok=True)
    df = load_data(train_csv)

    # If dataset contains duplicates (same id, different aspects) that's expected: each row is a sample.
    X = df['text'].astype(str).values
    y_aspect = df['aspectCategory'].astype(str).values
    y_polarity = df['polarity'].astype(str).values
    ids = df['id'].values

    # Split into train/dev (stratify by polarity maybe)
    X_tr, X_dev, aspect_tr, aspect_dev, pol_tr, pol_dev = train_test_split(
        X, y_aspect, y_polarity, test_size=0.2, random_state=42, stratify=y_polarity)

    # Label encoders (keep for mapping)
    le_aspect = LabelEncoder().fit(np.concatenate([aspect_tr, aspect_dev]))
    le_polarity = LabelEncoder().fit(np.concatenate([pol_tr, pol_dev]))

    y_aspect_tr = le_aspect.transform(aspect_tr)
    y_aspect_dev = le_aspect.transform(aspect_dev)

    y_pol_tr = le_polarity.transform(pol_tr)
    y_pol_dev = le_polarity.transform(pol_dev)

    # Train aspect classifier
    print("\nTraining aspect classifier...")
    if use_grid_search:
        aspect_pipe = grid_search_train(X_tr, aspect_tr)
    else:
        aspect_pipe = build_text_pipeline()
        aspect_pipe.fit(X_tr, aspect_tr)

    print("\nTraining sentiment classifier...")
    if use_grid_search:
        sentiment_pipe = grid_search_train(X_tr, pol_tr)
    else:
        sentiment_pipe = build_text_pipeline()
        sentiment_pipe.fit(X_tr, pol_tr)

    # Evaluate on dev set
    print("\nEvaluating on dev set...\n")
    aspect_pred_dev = aspect_pipe.predict(X_dev)
    pol_pred_dev = sentiment_pipe.predict(X_dev)

    print("Aspect classification report:")
    print(classification_report(aspect_dev, aspect_pred_dev, zero_division=0))
    print("Polarity classification report:")
    print(classification_report(pol_dev, pol_pred_dev, zero_division=0))

    # Overall — a prediction is correct only if both match.
    correct_both = (aspect_pred_dev == aspect_dev) & (pol_pred_dev == pol_dev)
    overall_f1 = f1_score((aspect_dev + '||' + pol_dev), (aspect_pred_dev + '||' + pol_pred_dev), average='macro')
    acc = correct_both.mean()
    print(f"Dev set: combined accuracy (both must match) = {acc:.4f}")
    print(f"Dev set: combined F1 (macro) using combined tokens = {overall_f1:.4f} (approx)")

    # Save models and encoders
    aspect_model_path = os.path.join(output_dir, 'aspect_model.joblib')
    sentiment_model_path = os.path.join(output_dir, 'sentiment_model.joblib')
    encoders_path = os.path.join(output_dir, 'label_encoders.joblib')

    joblib.dump(aspect_pipe, aspect_model_path)
    joblib.dump(sentiment_pipe, sentiment_model_path)
    joblib.dump({'le_aspect': le_aspect, 'le_polarity': le_polarity}, encoders_path)
    print(f"\nSaved aspect model -> {aspect_model_path}")
    print(f"Saved sentiment model -> {sentiment_model_path}")
    print(f"Saved encoders -> {encoders_path}")

    # If test file provided, produce predictions
    if test_csv is not None:
        df_test = pd.read_csv(test_csv)
        if 'text' not in df_test.columns or 'id' not in df_test.columns:
            raise ValueError("Test CSV must have columns 'id' and 'text'.")
        X_test = df_test['text'].astype(str).values
        ids_test = df_test['id'].values

        pred_aspect_test = aspect_pipe.predict(X_test)
        pred_pol_test = sentiment_pipe.predict(X_test)

        out_df = pd.DataFrame({
            'id': ids_test,
            'aspectCategory': pred_aspect_test,
            'polarity': pred_pol_test
        })
        out_csv = os.path.join(output_dir, 'test_pred.csv')
        out_df.to_csv(out_csv, index=False)
        print(f"\nSaved test predictions -> {out_csv}")
        print("Make sure to run your provided check_id.py and evaluate.py as required by the assignment.")

    return aspect_pipe, sentiment_pipe, (le_aspect, le_polarity)


def parse_args():
    p = argparse.ArgumentParser(description="Train logistic regression ABSA baseline")
    p.add_argument('--train', required=True, help='Path to training CSV with columns id,text,aspectCategory,polarity')
    p.add_argument('--test', required=False, help='(optional) test CSV with columns id,text to predict')
    p.add_argument('--out', default='models', help='Output folder to save models and predictions')
    p.add_argument('--no-grid', action='store_true', help='Disable GridSearch (faster)')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train_and_save(args.train, test_csv=args.test, output_dir=args.out, use_grid_search=(not args.no_grid))
