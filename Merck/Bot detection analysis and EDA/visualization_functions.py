# Visualization functions
def create_time_distribution_plot(df):
    """
    Create overlapping histograms for time distributions
    """
    plt.figure(figsize=(12, 8))
    
    plt.hist(df['sent_to_open'], bins=50, alpha=0.6, label="Sent to Open", color='purple')
    plt.hist(df['sent_to_click'], bins=50, alpha=0.7, label="Sent to Click", color='black')
    plt.hist(df['open_to_click'], bins=50, alpha=0.5, label="Open to Click", color='yellow')
    
    plt.xlabel("Time (seconds)", size=14)
    plt.ylabel("Count", size=14)
    plt.title("Email Engagement Time Distributions", size=16)
    plt.legend(loc='upper right')
    plt.yscale('log')  # Log scale for better visibility
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_correlation_heatmap(df):
    """
    Create correlation heatmap for numerical variables
    """
    # Select numerical columns for correlation
    numerical_cols = ['open_diff', 'click_diff', 'sent_to_open', 'sent_to_click', 'open_to_click', 'rn']
    
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[numerical_cols].corr()
    
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.3f')
    
    plt.title("Correlation Matrix - Email Engagement Metrics", size=16)
    plt.tight_layout()
    plt.show()

def create_browser_analysis_plot(df):
    """
    Create box plot for browser vs open-to-click time
    """
    plt.figure(figsize=(12, 8))
    
    # Filter extreme outliers for better visualization
    q95 = df['open_to_click'].quantile(0.95)
    filtered_df = df[df['open_to_click'] <= q95]
    
    sns.boxplot(x='browser', y='open_to_click', data=filtered_df)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Browser", size=14)
    plt.ylabel("Open to Click Time (seconds)", size=14)
    plt.title("Open-to-Click Time Distribution by Browser", size=16)
    
    plt.tight_layout()
    plt.show()

def create_pairplot_analysis(df):
    """
    Create pairplot for key engagement metrics
    """
    # Sample data for performance (if dataset is large)
    sample_size = min(10000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    cols = ['open_diff', 'click_diff', 'sent_to_open', 'sent_to_click', 'open_to_click']
    
    sns.pairplot(df_sample[cols], height=2.5, plot_kws={'alpha': 0.6})
    plt.suptitle("Pairplot - Email Engagement Metrics", y=1.02, size=16)
    plt.show()

# Execute visualization functions
create_time_distribution_plot(df)
create_correlation_heatmap(df)
create_browser_analysis_plot(df)
create_pairplot_analysis(df)