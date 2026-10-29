import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def load_train_data() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, 'wisesight_train.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def load_test_data() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, 'wisesight_test.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def load_sample_data() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, 'sample_customer_reviews.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def get_dataset_overview() -> dict:
    df_train = load_train_data()
    df_test = load_test_data()
    
    total_samples = len(df_train) + len(df_test)
    
    return {
        "total_samples": total_samples,
        "train_count": len(df_train),
        "test_count": len(df_test),
        "train_distribution": df_train['sentiment_th'].value_counts().to_dict() if not df_train.empty else {},
        "test_distribution": df_test['sentiment_th'].value_counts().to_dict() if not df_test.empty else {}
    }

