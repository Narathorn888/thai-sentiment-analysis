import os
import sys

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_train_data, load_test_data
from src.model import train_and_evaluate

def main():
    print("=" * 60)
    print("Training Baseline Model for Thai Sentiment Analysis")
    print("Dataset: Wisesight Sentiment Corpus (3 Classes)")
    print("=" * 60)
    
    df_train = load_train_data()
    df_test = load_test_data()
    
    if df_train.empty or df_test.empty:
        print("Error: Dataset CSVs not found. Run scripts/prepare_data.py first.")
        return
        
    print(f"Loaded Train Set: {len(df_train)} samples")
    print(f"Loaded Test Set: {len(df_test)} samples")
    print("\nClass distribution in Train:")
    for k, v in df_train['sentiment_th'].value_counts().items():
        print(f"  - {k}: {v} ({v/len(df_train):.1%})")
        
    print("\nTraining TF-IDF + Logistic Regression Model...")
    pipeline, results = train_and_evaluate(df_train, df_test, save_artifacts=True)
    
    print("\n" + "=" * 60)
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Macro F1-Score: {results['macro_f1']:.4f}")
    print(f"Weighted F1-Score: {results['weighted_f1']:.4f}")
    print("=" * 60)
    
    rep = results['classification_report']
    print("\nDetailed Class Metrics:")
    for cls in ['เชิงกลาง', 'เชิงบวก', 'เชิงลบ']:
        if cls in rep:
            p = rep[cls]['precision']
            r = rep[cls]['recall']
            f = rep[cls]['f1-score']
            s = rep[cls]['support']
            print(f"  {cls:8s} | Precision: {p:.2%} | Recall: {r:.2%} | F1: {f:.4f} | Support: {s}")
            
    print(f"\nModel and metrics successfully saved to models/ directory.")

if __name__ == '__main__':
    main()

