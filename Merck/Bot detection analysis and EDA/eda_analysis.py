# Exploratory Data Analysis
def perform_eda(df):
    """
    Perform comprehensive exploratory data analysis
    """
    # Unique value counts
    print("Unique values per column:")
    print(df.nunique())
    
    # Email frequency analysis
    print("\nTop email addresses by engagement frequency:")
    email_counts = df.groupby(['emailaddress']).size().sort_values(ascending=False)
    print(email_counts.head(20))
    
    # Time-based statistics
    time_columns = ['open_diff', 'click_diff', 'sent_to_open', 'sent_to_click', 'open_to_click']
    print(f"\nTime-based metrics statistics:")
    print(df[time_columns].describe())
    
    # Skewness and kurtosis analysis
    print(f"\nSkewness and Kurtosis for sent_to_click:")
    print(f"Skewness: {df['sent_to_click'].skew():.6f}")
    print(f"Kurtosis: {df['sent_to_click'].kurt():.6f}")
    
    return email_counts

# Perform EDA
email_frequency = perform_eda(df)