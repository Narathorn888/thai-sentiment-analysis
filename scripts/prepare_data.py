import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def process_file_pair(text_path, label_path):
    with open(text_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f]
    with open(label_path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f]
        
    label_map = {
        'pos': ('เชิงบวก', 'Positive'),
        'neu': ('เชิงกลาง', 'Neutral'),
        'neg': ('เชิงลบ', 'Negative')
    }
    
    rows = []
    for t, l in zip(texts, labels):
        if l in label_map and len(t) > 0:
            th_label, en_label = label_map[l]
            rows.append({
                'review_text': t,
                'sentiment_th': th_label,
                'sentiment_en': en_label,
                'raw_label': l
            })
            
    return pd.DataFrame(rows)

def main():
    print("Preparing and cleaning Wisesight Dataset...")
    train_txt = os.path.join(DATA_DIR, 'train.txt')
    train_lbl = os.path.join(DATA_DIR, 'train_label.txt')
    test_txt = os.path.join(DATA_DIR, 'test.txt')
    test_lbl = os.path.join(DATA_DIR, 'test_label.txt')
    
    if not (os.path.exists(train_txt) and os.path.exists(train_lbl)):
        raise FileNotFoundError("Raw Wisesight dataset files missing in data/ directory.")
        
    df_train = process_file_pair(train_txt, train_lbl)
    df_test = process_file_pair(test_txt, test_lbl)
    
    train_csv = os.path.join(DATA_DIR, 'wisesight_train.csv')
    test_csv = os.path.join(DATA_DIR, 'wisesight_test.csv')
    
    df_train.to_csv(train_csv, index=False, encoding='utf-8-sig')
    df_test.to_csv(test_csv, index=False, encoding='utf-8-sig')
    
    print(f"Saved Train CSV: {train_csv} ({len(df_train)} rows)")
    print(f"Saved Test CSV: {test_csv} ({len(df_test)} rows)")
    
    # Create a realistic sample CSV with diverse reviews (50 positive, 50 neutral, 50 negative)
    # for easy testing in the app's Batch Upload tab!
    sample_pos = df_test[df_test['sentiment_th'] == 'เชิงบวก'].head(30)
    sample_neu = df_test[df_test['sentiment_th'] == 'เชิงกลาง'].head(30)
    sample_neg = df_test[df_test['sentiment_th'] == 'เชิงลบ'].head(40)
    
    df_sample = pd.concat([sample_pos, sample_neu, sample_neg]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    df_sample = df_sample[['review_text', 'sentiment_th']].rename(columns={
        'review_text': 'ข้อความรีวิว',
        'sentiment_th': 'หมวดหมู่จริง'
    })
    
    sample_csv = os.path.join(DATA_DIR, 'sample_customer_reviews.csv')
    df_sample.to_csv(sample_csv, index=False, encoding='utf-8-sig')
    print(f"Saved Sample CSV for testing: {sample_csv} ({len(df_sample)} rows)")

if __name__ == '__main__':
    main()

