#!/usr/bin/env python3
"""
Email Bot Detection Analysis - Main Script
==========================================

This script performs comprehensive analysis of email engagement data
to detect potential bot behavior patterns.

Author: Data Science Team
Date: 2024
"""

# Setup proxy and install required packages (uncomment if needed)
import os
os.environ['http_proxy'] = "http://webproxy.merck.com:8080"     
os.environ['https_proxy'] = "http://webproxy.merck.com:8080"

# Import all required libraries
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import psycopg2
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def main():
    """
    Main function to execute the complete email bot detection analysis
    """
    print("🚀 Starting Email Bot Detection Analysis...")
    print("="*60)
    
    try:
        # Step 1: Fetch data from database
        print("📊 Step 1: Fetching data from database...")
        df = fetch_email_engagement_data()
        print(f"✅ Successfully loaded {len(df):,} records")
        
        # Step 2: Preprocess data
        print("\n🔧 Step 2: Preprocessing data...")
        df = preprocess_data(df)
        print("✅ Data preprocessing completed")
        
        # Step 3: Perform EDA
        print("\n📈 Step 3: Performing exploratory data analysis...")
        email_frequency = perform_eda(df)
        print("✅ EDA completed")
        
        # Step 4: Create visualizations
        print("\n📊 Step 4: Creating visualizations...")
        create_time_distribution_plot(df)
        create_correlation_heatmap(df)
        create_browser_analysis_plot(df)
        create_pairplot_analysis(df)
        print("✅ Visualizations completed")
        
        # Step 5: Bot detection analysis
        print("\n🤖 Step 5: Performing bot detection analysis...")
        suspicious_accounts, browser_bot_stats = analyze_bot_behavior(df)
        create_bot_detection_visualizations(df)
        print("✅ Bot detection analysis completed")
        
        # Step 6: Generate summary and export results
        print("\n📄 Step 6: Generating summary report and exporting results...")
        generate_summary_report(df, suspicious_accounts, browser_bot_stats)
        export_analysis_results(df, suspicious_accounts, browser_bot_stats)
        print("✅ Summary and export completed")
        
        print("\n🎉 Analysis pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        raise

# Include all function definitions here
# (Copy the functions from the other files or import them)

# Database connection function
def fetch_email_engagement_data():
    """Fetch email engagement data from the database"""
    sql_query = """
    SELECT s.sendid, s.subscriberkey, s.listid, s.batchid, s.emailaddress, 
           s.eventdate as sent_date, s.eventtype, o.eventdate as open_date,
           DATEDIFF(seconds, cast(sent_date as timestamp), cast(open_date as timestamp)) as open_diff,
           o.isunique, c.isuniqueforurl, c.gal_area_code, o.browser, 
           o.emailclient, o.operatingsystem, o.device,
           c.eventdate as click_date, c.eventtype, 
           DATEDIFF(seconds, cast(sent_date as timestamp), cast(click_date as timestamp)) as click_diff,
           ROW_NUMBER() OVER (PARTITION BY s.sendid, s.subscriberkey, s.listid, s.batchid, s.emailaddress 
                             ORDER BY s.emailaddress, open_diff ASC) as rn
    FROM cim_lc_pub.vw_c039_015_sent s
    INNER JOIN cim_lc_pub.vw_c039_011_opens o ON s.sendid = o.sendid AND s.subscriberkey = o.subscriberkey AND s.listid = o.listid AND s.batchid = o.batchid AND s.emailaddress = o.emailaddress
    INNER JOIN cim_lc_pub.vw_c039_004_clicks c ON s.sendid = c.sendid AND s.subscriberkey = c.subscriberkey AND s.listid = c.listid AND s.batchid = c.batchid AND s.emailaddress = c.emailaddress
    ORDER BY emailaddress, open_diff ASC;
    """
    
    db_string = "postgresql+psycopg2://cim_dwh_ro:PCimDwhRo1234$@awsapcimbirsp01.coaaq18eo1zb.ap-southeast-1.redshift.amazonaws.com:25881/apbirsp01"
    db = create_engine(db_string)
    return pd.read_sql(sql_query, db)

# Data preprocessing function
def preprocess_data(df):
    """Clean and preprocess the email engagement data"""
    date_columns = ['click_date', 'open_date', 'sent_date']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])
    
    df['open_to_click'] = (df['click_date'] - df['open_date']) / np.timedelta64(1, 's')
    df['sent_to_click'] = (df['click_date'] - df['sent_date']) / np.timedelta64(1, 's')
    df['sent_to_open'] = (df['open_date'] - df['sent_date']) / np.timedelta64(1, 's')
    
    categorical_columns = ['browser', 'emailclient', 'operatingsystem', 'device', 'gal_area_code']
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    df['potential_bot'] = df['open_diff'] < 10
    return df

# Add other function definitions here...
# (For brevity, I'm not copying all functions, but in practice you would)

if __name__ == "__main__":
    main()