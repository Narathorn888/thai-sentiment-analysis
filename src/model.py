import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from src.preprocessor import tokenize_thai

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_FILE = os.path.join(MODELS_DIR, "baseline_model.pkl")
METRICS_FILE = os.path.join(MODELS_DIR, "evaluation_results.json")

def create_baseline_pipeline(max_features=8000, ngram_range=(1, 2)):
    """
    Creates TF-IDF + Logistic Regression pipeline with balanced class weights.
    """
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            tokenizer=tokenize_thai,
            token_pattern=None,
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=3
        )),
        ('clf', LogisticRegression(
            class_weight='balanced',
            max_iter=500,
            random_state=42
        ))
    ])

def train_and_evaluate(train_df, test_df, save_artifacts=True):
    """
    Trains the baseline model on train_df and evaluates on test_df.
    Saves model and metrics to models/ directory.
    """
    X_train = train_df['review_text'].tolist()
    y_train = train_df['sentiment_th'].tolist()
    X_test = test_df['review_text'].tolist()
    y_test = test_df['sentiment_th'].tolist()
    
    pipeline = create_baseline_pipeline()
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)
    
    classes = ['เชิงกลาง', 'เชิงบวก', 'เชิงลบ']
    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average='macro'))
    weighted_f1 = float(f1_score(y_test, y_pred, average='weighted'))
    
    report_dict = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=classes).tolist()
    
    results = {
        "model_name": "TF-IDF + Logistic Regression (Balanced)",
        "tokenizer": "PyThaiNLP newmm",
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "classes": classes,
        "classification_report": report_dict,
        "confusion_matrix": cm,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_distribution": train_df['sentiment_th'].value_counts().to_dict(),
        "test_distribution": test_df['sentiment_th'].value_counts().to_dict()
    }
    
    if save_artifacts:
        joblib.dump(pipeline, MODEL_FILE)
        with open(METRICS_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
    return pipeline, results

def load_trained_model():
    """Loads the trained model pipeline."""
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            return None
    return None

def load_metrics():
    """Loads saved evaluation results."""
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    return None

def predict_single(model, text: str) -> dict:
    """
    Predicts sentiment and returns probabilities for all classes.
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "prediction": "เชิงกลาง",
            "confidence": 0.0,
            "probabilities": {"เชิงกลาง": 0.0, "เชิงบวก": 0.0, "เชิงลบ": 0.0}
        }
        
    probs = model.predict_proba([text])[0]
    classes = list(model.classes_)
    pred_idx = np.argmax(probs)
    
    prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, probs)}
    
    return {
        "prediction": classes[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "probabilities": prob_dict
    }

