# Bot Detection Analysis
def analyze_bot_behavior(df):
    """
    Analyze potential bot behavior patterns in email engagement
    """
    print("=== BOT DETECTION ANALYSIS ===\n")
    
    # Fast opens analysis (potential bots)
    fast_opens = df[df['open_diff'] < 10]
    print(f"Emails opened in <10 seconds: {len(fast_opens):,} ({len(fast_opens)/len(df)*100:.2f}%)")
    
    # Immediate clicks (open and click simultaneously)
    immediate_clicks = df[df['open_to_click'] <= 0]
    print(f"Immediate clicks (≤0 seconds): {len(immediate_clicks):,} ({len(immediate_clicks)/len(df)*100:.2f}%)")
    
    # Suspicious patterns by email
    suspicious_emails = df.groupby('emailaddress').agg({
        'open_diff': ['count', 'mean', 'std'],
        'open_to_click': 'mean',
        'potential_bot': 'sum'
    }).round(3)
    
    suspicious_emails.columns = ['total_engagements', 'avg_open_time', 'std_open_time', 
                                'avg_open_to_click', 'fast_open_count']
    
    # Filter for potentially suspicious behavior
    suspicious_criteria = (
        (suspicious_emails['fast_open_count'] > 5) |  # Multiple fast opens
        (suspicious_emails['avg_open_time'] < 5) |    # Consistently fast opens
        (suspicious_emails['std_open_time'] < 1)      # Very consistent timing
    )
    
    suspicious_accounts = suspicious_emails[suspicious_criteria].sort_values('fast_open_count', ascending=False)
    
    print(f"\nPotentially suspicious email accounts: {len(suspicious_accounts)}")
    print("\nTop 10 suspicious accounts:")
    print(suspicious_accounts.head(10))
    
    # Browser and device analysis for bots
    print("\n=== BOT PATTERNS BY BROWSER ===")
    browser_bot_stats = df.groupby('browser').agg({
        'potential_bot': ['count', 'sum'],
        'open_diff': 'mean'
    }).round(3)
    
    browser_bot_stats.columns = ['total_opens', 'fast_opens', 'avg_open_time']
    browser_bot_stats['bot_percentage'] = (browser_bot_stats['fast_opens'] / browser_bot_stats['total_opens'] * 100).round(2)
    browser_bot_stats = browser_bot_stats.sort_values('bot_percentage', ascending=False)
    
    print(browser_bot_stats)
    
    return suspicious_accounts, browser_bot_stats

def create_bot_detection_visualizations(df):
    """
    Create visualizations for bot detection analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Distribution of open times (log scale)
    axes[0, 0].hist(df['open_diff'], bins=100, alpha=0.7, color='skyblue')
    axes[0, 0].axvline(x=10, color='red', linestyle='--', label='10-second threshold')
    axes[0, 0].set_xlabel('Open Time (seconds)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of Email Open Times')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Open-to-click time distribution
    filtered_otc = df[(df['open_to_click'] >= -100) & (df['open_to_click'] <= 1000)]
    axes[0, 1].hist(filtered_otc['open_to_click'], bins=50, alpha=0.7, color='lightgreen')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', label='Immediate click')
    axes[0, 1].set_xlabel('Open-to-Click Time (seconds)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Distribution of Open-to-Click Times')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Bot percentage by browser
    browser_stats = df.groupby('browser')['potential_bot'].agg(['count', 'sum']).reset_index()
    browser_stats['bot_pct'] = (browser_stats['sum'] / browser_stats['count'] * 100).round(2)
    browser_stats = browser_stats.sort_values('bot_pct', ascending=True)
    
    axes[1, 0].barh(browser_stats['browser'], browser_stats['bot_pct'], color='orange', alpha=0.7)
    axes[1, 0].set_xlabel('Bot Percentage (%)')
    axes[1, 0].set_title('Potential Bot Percentage by Browser')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Engagement count vs bot behavior
    email_stats = df.groupby('emailaddress').agg({
        'potential_bot': ['count', 'sum']
    }).reset_index()
    email_stats.columns = ['emailaddress', 'total_engagements', 'bot_count']
    email_stats['bot_rate'] = email_stats['bot_count'] / email_stats['total_engagements']
    
    # Sample for plotting (avoid overplotting)
    sample_stats = email_stats.sample(n=min(1000, len(email_stats)), random_state=42)
    
    scatter = axes[1, 1].scatter(sample_stats['total_engagements'], 
                                sample_stats['bot_rate'], 
                                alpha=0.6, 
                                c=sample_stats['bot_count'], 
                                cmap='Reds')
    axes[1, 1].set_xlabel('Total Engagements')
    axes[1, 1].set_ylabel('Bot Rate')
    axes[1, 1].set_title('Bot Rate vs Total Engagements')
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 1], label='Bot Count')
    
    plt.tight_layout()
    plt.show()

# Execute bot detection analysis
suspicious_accounts, browser_bot_stats = analyze_bot_behavior(df)
create_bot_detection_visualizations(df)