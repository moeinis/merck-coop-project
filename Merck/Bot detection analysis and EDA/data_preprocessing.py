# Data preprocessing and feature engineering
def preprocess_data(df):
    """
    Clean and preprocess the email engagement data
    """
    # Convert date columns to datetime
    date_columns = ['click_date', 'open_date', 'sent_date']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])
    
    # Create time difference features (in seconds)
    df['open_to_click'] = (df['click_date'] - df['open_date']) / np.timedelta64(1, 's')
    df['sent_to_click'] = (df['click_date'] - df['sent_date']) / np.timedelta64(1, 's')
    df['sent_to_open'] = (df['open_date'] - df['sent_date']) / np.timedelta64(1, 's')
    
    # Convert categorical columns
    categorical_columns = ['browser', 'emailclient', 'operatingsystem', 'device', 'gal_area_code']
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # Create bot detection flag for very fast opens (< 10 seconds)
    df['potential_bot'] = df['open_diff'] < 10
    
    return df

# Preprocess the data
df = preprocess_data(df)

# Display basic information about the dataset
print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes)
print("\nBasic statistics:")
print(df.describe())