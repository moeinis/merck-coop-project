# Summary and Export Functions
def generate_summary_report(df, suspicious_accounts, browser_bot_stats):
    """
    Generate a comprehensive summary report
    """
    print("="*60)
    print("EMAIL ENGAGEMENT ANALYSIS SUMMARY REPORT")
    print("="*60)
    
    # Dataset overview
    print(f"\n📊 DATASET OVERVIEW")
    print(f"Total records: {len(df):,}")
    print(f"Unique email addresses: {df['emailaddress'].nunique():,}")
    print(f"Date range: {df['sent_date'].min()} to {df['sent_date'].max()}")
    print(f"Unique campaigns (sendid): {df['sendid'].nunique():,}")
    
    # Engagement metrics
    print(f"\n⏱️ ENGAGEMENT TIMING METRICS")
    print(f"Average open time: {df['open_diff'].mean():.2f} seconds")
    print(f"Median open time: {df['open_diff'].median():.2f} seconds")
    print(f"Average click time: {df['click_diff'].mean():.2f} seconds")
    print(f"Median click time: {df['click_diff'].median():.2f} seconds")
    print(f"Average open-to-click: {df['open_to_click'].mean():.2f} seconds")
    
    # Bot detection summary
    print(f"\n🤖 BOT DETECTION SUMMARY")
    fast_opens = (df['open_diff'] < 10).sum()
    immediate_clicks = (df['open_to_click'] <= 0).sum()
    print(f"Potential bot opens (<10s): {fast_opens:,} ({fast_opens/len(df)*100:.2f}%)")
    print(f"Immediate clicks (≤0s): {immediate_clicks:,} ({immediate_clicks/len(df)*100:.2f}%)")
    print(f"Suspicious email accounts: {len(suspicious_accounts):,}")
    
    # Top browsers and devices
    print(f"\n🌐 TOP BROWSERS")
    top_browsers = df['browser'].value_counts().head(5)
    for browser, count in top_browsers.items():
        print(f"  {browser}: {count:,} ({count/len(df)*100:.1f}%)")
    
    print(f"\n💻 TOP DEVICES")
    top_devices = df['device'].value_counts().head(5)
    for device, count in top_devices.items():
        print(f"  {device}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Geographic distribution
    print(f"\n🌍 GEOGRAPHIC DISTRIBUTION")
    top_regions = df['gal_area_code'].value_counts().head(5)
    for region, count in top_regions.items():
        print(f"  {region}: {count:,} ({count/len(df)*100:.1f}%)")
    
    print("="*60)

def export_analysis_results(df, suspicious_accounts, browser_bot_stats, output_dir='./output/'):
    """
    Export analysis results to CSV files
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Export main dataset (sample for performance)
    sample_df = df.sample(n=min(50000, len(df)), random_state=42)
    sample_df.to_csv(f'{output_dir}email_engagement_sample.csv', index=False)
    print(f"✅ Exported sample dataset to {output_dir}email_engagement_sample.csv")
    
    # Export suspicious accounts
    suspicious_accounts.to_csv(f'{output_dir}suspicious_accounts.csv')
    print(f"✅ Exported suspicious accounts to {output_dir}suspicious_accounts.csv")
    
    # Export browser bot statistics
    browser_bot_stats.to_csv(f'{output_dir}browser_bot_statistics.csv')
    print(f"✅ Exported browser statistics to {output_dir}browser_bot_statistics.csv")
    
    # Export summary statistics
    summary_stats = {
        'total_records': len(df),
        'unique_emails': df['emailaddress'].nunique(),
        'unique_campaigns': df['sendid'].nunique(),
        'avg_open_time': df['open_diff'].mean(),
        'avg_click_time': df['click_diff'].mean(),
        'potential_bot_percentage': (df['potential_bot'].sum() / len(df) * 100),
        'suspicious_accounts_count': len(suspicious_accounts)
    }
    
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv(f'{output_dir}summary_statistics.csv', index=False)
    print(f"✅ Exported summary statistics to {output_dir}summary_statistics.csv")

# Generate final summary and export results
generate_summary_report(df, suspicious_accounts, browser_bot_stats)
export_analysis_results(df, suspicious_accounts, browser_bot_stats)

print("\n🎉 Analysis complete! All results have been exported.")