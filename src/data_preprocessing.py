import json
import pandas as pd

def load_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def preprocess_data(df):
    # Filter by status and select relevant columns
    df = df[df['status'] == 'A']
    df = df[['description', 'class_id']]

    # Filter out non-integer and invalid (0) class_id values
    df = df[df['class_id'].apply(lambda x: x.isdigit() and int(x) > 0)]  # Keep only rows where class_id is a positive integer

    # Convert class_ids to integers
    df['class_id'] = df['class_id'].astype(int)

    # Reset index after filtering
    df = df.reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    df = load_data('./data/idmanual.json')
    df = preprocess_data(df)
    df.to_csv('./data/preprocessed_data.csv', index=False)
