import json
import pandas as pd

def load_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def preprocess_data(df):
    df = df[df['status'] == 'A']
    df = df[['description', 'class_id']]
    return df

if __name__ == "__main__":
    df = load_data('./data/idmanual.json')
    df = preprocess_data(df)
    df.to_csv('./data/preprocessed_data.csv', index=False)
