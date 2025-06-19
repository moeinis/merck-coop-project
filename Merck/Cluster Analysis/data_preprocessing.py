# Data Loading and Preprocessing
import pandas as pd
import numpy as np

def load_and_explore_data(file_path='dataframe_for_clustering.csv'):
    """Load data and perform initial exploration"""
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Unique values per column:\n{df.nunique()}")
    print(f"Data types:\n{df.dtypes}")
    
    return df

def create_balanced_dataset(df, sample_size=800000, random_state=1):
    """Create a balanced dataset for analysis"""
    # Separate unknown and labeled data
    df_unknown = df[df['final_label'].astype(str) == 'unknown']
    df_unknown_sample = df_unknown.sample(sample_size, random_state=random_state)
    df_unknown_sample['final_label'] = df_unknown_sample['final_label'].replace('unknown', 'False')
    
    # Get all True/False labeled data
    df_labeled = df[df['final_label'].astype(str) != 'unknown']
    
    # Combine datasets
    df_balanced = pd.concat([df_labeled, df_unknown_sample], ignore_index=True)
    
    print(f"Balanced dataset shape: {df_balanced.shape}")
    print(f"Label distribution:\n{df_balanced['final_label'].value_counts()}")
    
    return df_balanced

def preprocess_features(df):
    """Preprocess features for modeling"""
    # Remove unnecessary columns if they exist
    columns_to_remove = ['emailaddress', 'Unnamed: 0']
    for col in columns_to_remove:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    # Set proper data types
    categorical_columns = ['isuniqueforurl', 'isunique', 'browser', 'emailclient', 
                          'operatingsystem', 'device', 'is_immediate', 'has_trap_link']
    
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    return df

def create_one_hot_encoding(df, categorical_features=None):
    """Create one-hot encoded features"""
    if categorical_features is None:
        categorical_features = ['isunique', 'isuniqueforurl', 'emailclient', 
                               'browser', 'device', 'operatingsystem']
    
    # Filter existing columns
    existing_features = [col for col in categorical_features if col in df.columns]
    
    one_hot_data = pd.get_dummies(df[existing_features], drop_first=True, dtype=float)
    
    print(f"One-hot encoded shape: {one_hot_data.shape}")
    print(f"Features: {list(one_hot_data.columns)}")
    
    return one_hot_data

def create_test_dataset(df, sample_size=100000, random_state=1):
    """Create a smaller balanced test dataset"""
    df_unknown = df[df['final_label'].astype(str) == 'unknown']
    df_unknown_sample = df_unknown.sample(sample_size//2, random_state=random_state)
    df_unknown_sample['final_label'] = df_unknown_sample['final_label'].replace('unknown', 'False')
    
    df_true = df[df['final_label'].astype(str) == 'True']
    df_true_sample = df_true.sample(sample_size//2, random_state=random_state)
    
    df_test = pd.concat([df_true_sample, df_unknown_sample], ignore_index=True)
    df_test = df_test.sample(sample_size, random_state=random_state)
    
    print(f"Test dataset shape: {df_test.shape}")
    print(f"Test label distribution:\n{df_test['final_label'].value_counts()}")
    
    return df_test

if __name__ == "__main__":
    # Example usage
    df = load_and_explore_data()
    df_balanced = create_balanced_dataset(df)
    df_processed = preprocess_features(df_balanced)
    one_hot_features = create_one_hot_encoding(df_processed)
