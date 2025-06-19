# Email Bot Activity Analysis and Anomaly Detection
# =====================================================

# Setup and Imports
import os
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import psycopg2
import plotly.offline as pyo
import plotly.graph_objs as go
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import IsolationForest
from pycaret.utils import enable_colab
from pycaret.anomaly import *
from pycaret.classification import *

# Environment Setup
os.environ['http_proxy'] = "http://webproxy.merck.com:8080"     
os.environ['https_proxy'] = "http://webproxy.merck.com:8080"

# Display settings
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
plt.style.use('default')
pyo.init_notebook_mode()
enable_colab()

# Data Loading and Initial Exploration
# ====================================

def load_and_explore_data():
    """Load and perform initial exploration of the datasets"""
    
    # Load bot email data
    df1 = pd.read_csv('DIRECT_INPUT_BOT_EMAIL.csv')
    print(f"Bot Email Data Shape: {df1.shape}")
    print(f"Bot Activity Distribution:\n{df1['BOT_ACTIVITY'].value_counts()}")
    
    # Load SFMC action data  
    df2 = pd.read_csv('DIRECT_INPUT_SFMC_F_ME_ACTION.csv')
    print(f"SFMC Action Data Shape: {df2.shape}")
    
    # Check for duplicates
    duplicates = df1[df1.duplicated()]
    print(f"Duplicates found: {len(duplicates)}")
    
    return df1, df2

def preprocess_data(df1, df2):
    """Clean and preprocess the data"""
    
    # Remove rows starting with 'a1t' in SentId
    df1_clean = df1[~df1['SentId'].str.startswith('a1t')]
    
    # Create updated SentId by removing middle part
    df1_clean['SentId_updated'] = df1_clean['SentId'].str.split('|').apply(
        lambda x: '|'.join([x[0], x[2], x[3]]) if len(x) >= 4 else x[0]
    )
    
    # Convert date columns
    for col in ['ME Event Date', 'ME Event Time', '%Calendar Key']:
        if col in df2.columns:
            df2[col] = pd.to_datetime(df2[col], errors='coerce')
    
    return df1_clean, df2

def join_datasets(df1, df2):
    """Join the datasets on common keys"""
    
    # Primary join on ClickId_ME and EIS_ID
    joined_table1 = pd.merge(df1, df2, how='inner', left_on='ClickId_ME', right_on='EIS_ID')
    
    # Secondary join on SentId_updated and User Activity Key
    joined_table2 = pd.merge(df1, df2, how='inner', 
                            left_on='SentId_updated', right_on='%User Activity Key')
    
    print(f"First join shape: {joined_table1.shape}")
    print(f"Second join shape: {joined_table2.shape}")
    
    return joined_table1, joined_table2

def clean_joined_data(joined_table):
    """Clean the joined dataset and remove unnecessary columns"""
    
    # Remove unnecessary columns
    cols_to_drop = [
        'EMAIL_TYPE', 'ClickId_AE', 'ClickId_ME', 'EIS_ID_x', 'NON_BOT',
        '%Country Mailing Key', '%User Activity Key', '%Country Key', 
        '%Mailing Key', '%MDM Key', '%Campaign Key', 'Subscriber_List ID',
        'Triggered Email Key', 'ME Bounce Reason', 'ME Unsubscribe Reason', 
        'EIS_ID_y', '%Journey Activity Key', 'ME Bounce Category'
    ]
    
    # Drop columns that exist
    existing_cols = [col for col in cols_to_drop if col in joined_table.columns]
    final_table = joined_table.drop(existing_cols, axis=1)
    
    # Remove rows with missing browser/client/OS/device info
    required_cols = ['ME Browser', 'ME Email Client', 'ME OperatingSystem', 'ME Device']
    existing_required = [col for col in required_cols if col in final_table.columns]
    
    if existing_required:
        final_table = final_table.dropna(subset=existing_required)
    
    return final_table

# Feature Engineering
# ===================

def create_time_features(df):
    """Create time-based features for analysis"""
    
    # Ensure datetime columns
    date_cols = ['ME Event Date_FROM_SENT', 'ME Event Date_FROM_CLICK_OR_OPEN']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Create time difference features
    if 'ME Event Date_FROM_SENT' in df.columns and 'ME Event Date_FROM_CLICK_OR_OPEN' in df.columns:
        df['Sent_to_OpenClick'] = (df['ME Event Date_FROM_CLICK_OR_OPEN'] - 
                                  df['ME Event Date_FROM_SENT']).dt.total_seconds()
    
    # Add reference time features
    reference_time = pd.to_datetime('2021-03-01 00:00:00')
    
    if 'ME Event Date_FROM_CLICK_OR_OPEN' in df.columns:
        df['reference_time_to_Open'] = (df['ME Event Date_FROM_CLICK_OR_OPEN'] - 
                                       reference_time).dt.total_seconds()
    
    # Create immediate action indicators
    if 'Sent_to_OpenClick' in df.columns:
        df['is_immediate_action'] = np.where(df['Sent_to_OpenClick'] < 30, 1, 0)
    
    return df

def separate_event_types(df):
    """Separate data by event types (Open, Click, Sent)"""
    
    event_type_col = 'ME Event Type_FROM_CLICK_OR_OPEN'
    if event_type_col not in df.columns:
        event_type_col = 'ME Event Type'
    
    if event_type_col in df.columns:
        df_open = df[df[event_type_col] == 'Open'].copy()
        df_click = df[df[event_type_col] == 'Click'].copy()
        df_sent = df[df[event_type_col] == 'Sent'].copy()
        
        return df_open, df_click, df_sent
    
    return df, pd.DataFrame(), pd.DataFrame()

# Anomaly Detection Models
# ========================

def prepare_anomaly_data(df, target_col='BOT_ACTIVITY_FROM_CLICK_OR_OPEN'):
    """Prepare data for anomaly detection"""
    
    # Filter out German model predictions if specified
    df_filtered = df.copy()
    if target_col in df.columns:
        df_filtered = df_filtered[~df_filtered[target_col].isin(['Bot_open_model', 'Bot_click_model'])]
        
        # Convert to binary: false -> 0, honeypot variants -> 1
        df_filtered[target_col] = df_filtered[target_col].replace({
            'false': 0,
            'Bot_click_Honeypot': 1,
            'Bot_click_Honeypot_around': 1
        })
    
    return df_filtered

def run_isolation_forest(df, features, target_col, contamination=0.05):
    """Run Isolation Forest for anomaly detection"""
    
    # Prepare features
    feature_cols = [col for col in features if col in df.columns]
    X = df[feature_cols].copy()
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    else:
        X_encoded = X
    
    # Remove any remaining non-numeric columns
    X_encoded = X_encoded.select_dtypes(include=[np.number])
    
    if target_col in df.columns:
        y = df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42
        )
        
        # Fit on majority class (normal behavior)
        X_train_normal = X_train[y_train == 0]
        
        # Train model
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(X_train_normal)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Convert predictions: 1 -> normal (0), -1 -> anomaly (1)
        y_pred_binary = np.where(y_pred == 1, 0, 1)
        
        # Calculate F1 score
        f1 = f1_score(y_test, y_pred_binary)
        
        return model, f1, y_test, y_pred_binary
    
    return None, None, None, None

def run_pycaret_anomaly(df, features, target_col):
    """Run PyCaret anomaly detection"""
    
    try:
        # Select features for anomaly detection
        feature_cols = [col for col in features if col in df.columns]
        df_subset = df[feature_cols + [target_col]].copy()
        
        # Setup PyCaret
        anomaly_setup = setup(
            data=df_subset,
            session_id=123,
            silent=True
        )
        
        # Create model
        iforest = create_model('iforest')
        
        # Assign anomaly labels
        result = assign_model(iforest, score=True)
        
        # Create confusion matrix
        if target_col in result.columns:
            cm = pd.crosstab(
                result['Anomaly'], 
                result[target_col], 
                rownames=['Predicted'], 
                colnames=['Actual']
            )
            return iforest, result, cm
        
    except Exception as e:
        print(f"PyCaret anomaly detection failed: {e}")
        return None, None, None
    
    return None, None, None

# Classification Models
# =====================

def run_pycaret_classification(df, features, target_col):
    """Run PyCaret classification"""
    
    try:
        # Prepare data
        feature_cols = [col for col in features if col in df.columns]
        ignore_cols = ['ME Action Email', 'SentId'] if 'ME Action Email' in df.columns else []
        
        # Setup PyCaret
        clf_setup = setup(
            data=df,
            target=target_col,
            ignore_features=ignore_cols,
            ignore_low_variance=True,
            combine_rare_levels=True,
            remove_multicollinearity=True,
            data_split_stratify=True,
            session_id=123,
            silent=True
        )
        
        # Compare models
        best_models = compare_models(
            include=['rf', 'lightgbm', 'dt', 'knn', 'lr'],
            sort='F1',
            n_select=3
        )
        
        return best_models[0], clf_setup
        
    except Exception as e:
        print(f"PyCaret classification failed: {e}")
        return None, None

# Visualization and Analysis
# ==========================

def plot_time_distributions(df):
    """Plot distributions of time-based features"""
    
    time_cols = ['Sent_to_OpenClick', 'reference_time_to_Open']
    existing_time_cols = [col for col in time_cols if col in df.columns]
    
    if existing_time_cols:
        plt.figure(figsize=(12, 6))
        
        for i, col in enumerate(existing_time_cols, 1):
            plt.subplot(1, len(existing_time_cols), i)
            plt.hist(df[col].dropna(), bins=50, alpha=0.7)
            plt.title(f'Distribution of {col}')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()

def analyze_bot_patterns(df, target_col='BOT_ACTIVITY_FROM_CLICK_OR_OPEN'):
    """Analyze patterns in bot activity"""
    
    if target_col in df.columns:
        print("Bot Activity Distribution:")
        print(df[target_col].value_counts())
        print(f"\nBot Activity Percentage:")
        print(df[target_col].value_counts(normalize=True) * 100)
        
        # Analyze by event type if available
        event_col = 'ME Event Type_FROM_CLICK_OR_OPEN'
        if event_col in df.columns:
            print(f"\nBot Activity by Event Type:")
            crosstab = pd.crosstab(df[event_col], df[target_col])
            print(crosstab)

# Main Analysis Pipeline
# =====================

def main_analysis():
    """Main analysis pipeline"""
    
    print("=== Email Bot Activity Analysis ===\n")
    
    # 1. Load and explore data
    print("1. Loading and exploring data...")
    df1, df2 = load_and_explore_data()
    
    # 2. Preprocess data
    print("\n2. Preprocessing data...")
    df1_clean, df2_clean = preprocess_data(df1, df2)
    
    # 3. Join datasets
    print("\n3. Joining datasets...")
    joined_table1, joined_table2 = join_datasets(df1_clean, df2_clean)
    
    # 4. Clean joined data
    print("\n4. Cleaning joined data...")
    final_data = clean_joined_data(joined_table1)
    
    # 5. Feature engineering
    print("\n5. Creating time features...")
    final_data = create_time_features(final_data)
    
    # 6. Separate by event types
    print("\n6. Separating by event types...")
    df_open, df_click, df_sent = separate_event_types(final_data)
    
    # 7. Analyze patterns
    print("\n7. Analyzing bot patterns...")
    analyze_bot_patterns(final_data)
    
    # 8. Anomaly Detection
    print("\n8. Running anomaly detection...")
    
    # Prepare features for modeling
    numeric_features = ['Sent_to_OpenClick', 'reference_time_to_Open', 'is_immediate_action']
    categorical_features = ['ME Browser', 'ME Email Client', 'ME OperatingSystem', 'ME Device']
    all_features = numeric_features + categorical_features
    
    # Filter features that exist in data
    available_features = [f for f in all_features if f in final_data.columns]
    
    if len(available_features) > 0:
        # Prepare data for anomaly detection
        anomaly_data = prepare_anomaly_data(final_data)
        
        # Run Isolation Forest
        if len(anomaly_data) > 100:  # Ensure sufficient data
            model, f1_score, y_test, y_pred = run_isolation_forest(
                anomaly_data, available_features, 'BOT_ACTIVITY_FROM_CLICK_OR_OPEN'
            )
            
            if f1_score is not None:
                print(f"Isolation Forest F1 Score: {f1_score:.3f}")
        
        # Run PyCaret anomaly detection
        pycaret_model, pycaret_result, confusion_matrix = run_pycaret_anomaly(
            anomaly_data, available_features, 'BOT_ACTIVITY_FROM_CLICK_OR_OPEN'
        )
        
        if confusion_matrix is not None:
            print("\nPyCaret Anomaly Detection Confusion Matrix:")
            print(confusion_matrix)
    
    # 9. Classification
    print("\n9. Running classification...")
    
    if 'BOT_ACTIVITY_FROM_CLICK_OR_OPEN' in final_data.columns:
        best_classifier, setup_info = run_pycaret_classification(
            final_data, available_features, 'BOT_ACTIVITY_FROM_CLICK_OR_OPEN'
        )
        
        if best_classifier is not None:
            print(f"Best classifier: {type(best_classifier).__name__}")
    
    # 10. Visualizations
    print("\n10. Creating visualizations...")
    plot_time_distributions(final_data)
    
    print("\n=== Analysis Complete ===")
    
    return final_data, df_open, df_click, df_sent

# Additional utility functions
# ===========================

def save_results(df, filename_prefix='email_bot_analysis'):
    """Save analysis results to CSV"""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{filename_prefix}_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"Results saved to: {filename}")

def create_summary_report(df, target_col='BOT_ACTIVITY_FROM_CLICK_OR_OPEN'):
    """Create a summary report of the analysis"""
    
    report = {
        'total_records': len(df),
        'date_range': None,
        'bot_activity_distribution': None,
        'feature_summary': {}
    }
    
    # Date range
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    if date_cols:
        for col in date_cols:
            if df[col].dtype == 'datetime64[ns]':
                report['date_range'] = f"{df[col].min()} to {df[col].max()}"
                break
    
    # Bot activity distribution
    if target_col in df.columns:
        report['bot_activity_distribution'] = df[target_col].value_counts().to_dict()
    
    # Feature summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in df.columns:
            report['feature_summary'][col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'missing_pct': float(df[col].isnull().mean() * 100)
            }
    
    return report

# Run the analysis
if __name__ == "__main__":
    try:
        final_data, df_open, df_click, df_sent = main_analysis()
        
        # Generate summary report
        summary = create_summary_report(final_data)
        print("\n=== Summary Report ===")
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        # Save results
        save_results(final_data)
        
    except Exception as e:
        print(f"Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
