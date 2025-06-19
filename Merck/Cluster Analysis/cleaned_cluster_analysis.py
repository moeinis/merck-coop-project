"""
Cluster Analysis and Classification Pipeline
Cleaned and consolidated version for GitHub
"""

# Core imports
import os
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt

# Database and SQL
from sqlalchemy import create_engine, text
import psycopg2

# Plotting libraries
import plotly.offline as pyo
import plotly.graph_objs as go

# Machine Learning - Scikit-learn
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn import metrics, tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.decomposition import PCA
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

# Advanced clustering and dimensionality reduction
import umap
import hdbscan

# Visualization tools
from yellowbrick.model_selection import FeatureImportances
from IPython.display import Image as PImage
from PIL import Image, ImageDraw, ImageFont
import graphviz

# PyCaret
from pycaret.utils import enable_colab
from pycaret.classification import *
from pycaret.clustering import *
from pycaret.anomaly import *

# Configuration
def setup_environment():
    """Configure environment settings"""
    # Uncomment if using Merck proxy
    # os.environ['http_proxy'] = "http://webproxy.merck.com:8080"     
    # os.environ['https_proxy'] = "http://webproxy.merck.com:8080"
    
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    
    pyo.init_notebook_mode()
    enable_colab()
    plt.style.use('default')
    sns.set(style='white', rc={'figure.figsize': (10, 8)})

# Data Loading and Preprocessing
def load_data(filepath='dataframe_for_clustering.csv'):
    """Load and explore dataset"""
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Unique values:\n{df.nunique()}")
    return df

def create_balanced_dataset(df, sample_size=800000, random_state=1):
    """Create balanced dataset for analysis"""
    # Remove columns that might cause issues
    columns_to_remove = ['emailaddress', 'Unnamed: 0']
    for col in columns_to_remove:
        if col in df.columns:
            del df[col]
    
    # Create balanced dataset
    df_unknown = df[df['final_label'].astype(str) == 'unknown']
    df_unknown_sample = df_unknown.sample(sample_size, random_state=random_state)
    df_unknown_sample['final_label'] = df_unknown_sample['final_label'].replace('unknown', 'False')
    
    df_labeled = df[df['final_label'].astype(str) != 'unknown']
    df_balanced = pd.concat([df_labeled, df_unknown_sample])
    
    print(f"Balanced dataset shape: {df_balanced.shape}")
    print(f"Label distribution:\n{df_balanced['final_label'].value_counts()}")
    return df_balanced

def prepare_features(df):
    """Prepare features for modeling"""
    # One-hot encoding for categorical features
    categorical_features = ['isunique', 'isuniqueforurl', 'emailclient', 'browser', 'device', 'operatingsystem']
    existing_features = [col for col in categorical_features if col in df.columns]
    
    one_hot_data = pd.get_dummies(df[existing_features], drop_first=True, dtype=float)
    print(f"One-hot encoded shape: {one_hot_data.shape}")
    return one_hot_data

# Decision Tree Analysis
def extract_tree_rules(tree_model, feature_names, class_names):
    """Extract human-readable rules from decision tree"""
    tree_ = tree_model.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []
    
    def recurse(node, path, paths):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]
            
    recurse(0, path, paths)
    
    # Sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]
    
    rules = []
    for path in paths:
        rule = "if "
        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        
        if class_names is None:
            rule += "response: " + str(np.round(path[-1][0][0][0], 3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {np.round(100.0*classes[l]/np.sum(classes), 2)}%)"
        rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]
        
    return rules

def run_decision_tree_analysis(one_hot_data, df_balanced):
    """Run decision tree classification analysis"""
    print("="*50)
    print("DECISION TREE ANALYSIS")
    print("="*50)
    
    # Prepare data
    X, y = one_hot_data, df_balanced['final_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    
    # Train model
    dt = DecisionTreeClassifier(max_depth=5)
    dt.fit(X_train, y_train)
    
    print(f"Train score: {dt.score(X_train, y_train):.4f}")
    print(f"Test score: {dt.score(X_test, y_test):.4f}")
    
    # Feature importance visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    tree_viz = FeatureImportances(dt)
    tree_viz.fit(X_train, y_train)
    plt.show()
    
    # Tree visualization
    dot_data = tree.export_graphviz(
        dt, out_file=None, feature_names=one_hot_data.columns,
        filled=True, rounded=True, special_characters=True
    )
    graph = graphviz.Source(dot_data)
    
    # Extract and display rules
    rules = extract_tree_rules(dt, list(X.columns), ['True', 'False'])
    print("\nTop Decision Tree Rules:")
    for i, rule in enumerate(rules[:10], 1):
        print(f"{i}. {rule}")
    
    return dt, rules

# PyCaret Classification
def run_pycaret_classification(df_test):
    """Run PyCaret classification pipeline"""
    print("="*50)
    print("PYCARET CLASSIFICATION")
    print("="*50)
    
    # Setup experiment
    exp_clf = setup(
        data=df_test,
        target='final_label',
        session_id=123,
        ignore_low_variance=True,
        combine_rare_levels=True,
        remove_multicollinearity=True,
        use_gpu=True
    )
    
    # Compare models
    best_models = compare_models()
    
    # Create and tune specific models
    rf = create_model('rf')
    dt = create_model('dt')
    
    # Tune the best model (decision tree in this case)
    tuned_dt = tune_model(dt)
    
    # Plot model performance
    plot_model(rf, plot='auc')
    plot_model(tuned_dt, plot='pr')
    plot_model(dt, plot='confusion_matrix')
    
    # Make predictions
    predictions = predict_model(dt)
    
    # Finalize model
    final_model = finalize_model(tuned_dt)
    
    return final_model, predictions

# Clustering Analysis
def run_clustering_analysis(df_test):
    """Run comprehensive clustering analysis"""
    print("="*50)
    print("CLUSTERING ANALYSIS")
    print("="*50)
    
    # Sample data for clustering
    df_sample = df_test.sample(200000, random_state=1) if len(df_test) > 200000 else df_test
    
    # Setup clustering experiment
    exp_clu = setup(df_sample, ignore_features=['final_label'], session_id=123)
    
    # K-means clustering
    print("\nK-MEANS CLUSTERING:")
    kmeans = create_model('kmeans', ground_truth='final_label', num_clusters=2)
    
    # Plot K-means results
    plot_model(kmeans)
    plot_model(kmeans, plot='elbow')
    plot_model(kmeans, plot='silhouette')
    plot_model(kmeans, plot='distribution')
    
    # Assign clusters and evaluate
    kmeans_results = assign_model(kmeans)
    confusion_matrix = pd.crosstab(
        kmeans_results['final_label'], 
        kmeans_results['Cluster'], 
        rownames=['Actual'], 
        colnames=['Predicted']
    )
    print("K-means Confusion Matrix:")
    print(confusion_matrix)
    
    # DBSCAN clustering
    print("\nDBSCAN CLUSTERING:")
    dbscan = create_model('dbscan')
    plot_model(dbscan)
    plot_model(dbscan, plot='distribution')
    dbscan_results = assign_model(dbscan)
    
    # K-modes clustering
    print("\nK-MODES CLUSTERING:")
    kmodes = create_model('kmodes', num_clusters=3)
    plot_model(kmodes)
    plot_model(kmodes, plot='elbow')
    kmodes_results = assign_model(kmodes)
    
    # Hierarchical clustering
    print("\nHIERARCHICAL CLUSTERING:")
    hclust = create_model('hclust', num_clusters=3)
    plot_model(hclust)
    hclust_results = assign_model(hclust)
    
    return {
        'kmeans': (kmeans, kmeans_results),
        'dbscan': (dbscan, dbscan_results), 
        'kmodes': (kmodes, kmodes_results),
        'hclust': (hclust, hclust_results)
    }

# UMAP and Advanced Clustering
def run_umap_analysis(one_hot_data):
    """Run UMAP dimensionality reduction and clustering"""
    print("="*50)
    print("UMAP ANALYSIS")
    print("="*50)
    
    # Standard UMAP embedding
    standard_embedding = umap.UMAP(random_state=42).fit_transform(one_hot_data)
    plt.figure(figsize=(10, 8))
    plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], s=0.1, cmap='Spectral')
    plt.title("Standard UMAP Embedding")
    plt.show()
    
    # Clusterable embedding
    clusterable_embedding = umap.UMAP(
        n_neighbors=30, min_dist=0.0, n_components=2, random_state=42
    ).fit_transform(one_hot_data)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1], s=0.1, cmap='Spectral')
    plt.title("Clusterable UMAP Embedding")
    plt.show()
    
    # K-means on UMAP embedding
    kmeans_labels = cluster.KMeans(n_clusters=10).fit_predict(one_hot_data)
    plt.figure(figsize=(10, 8))
    plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=kmeans_labels, s=0.1, cmap='Spectral')
    plt.title("K-means Clustering on UMAP")
    plt.show()
    
    # HDBSCAN clustering
    lowd_data = PCA(n_components=5).fit_transform(one_hot_data)
    hdbscan_labels = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=500).fit_predict(lowd_data)
    
    clustered = (hdbscan_labels >= 0)
    plt.figure(figsize=(10, 8))
    plt.scatter(standard_embedding[~clustered, 0], standard_embedding[~clustered, 1],
                c=(0.5, 0.5, 0.5), s=0.1, alpha=0.5)
    plt.scatter(standard_embedding[clustered, 0], standard_embedding[clustered, 1],
                c=hdbscan_labels[clustered], s=0.1, cmap='Spectral')
    plt.title("HDBSCAN Clustering")
    plt.show()
    
    return standard_embedding, clusterable_embedding, kmeans_labels, hdbscan_labels

# Anomaly Detection
def run_anomaly_detection(df_test):
    """Run anomaly detection analysis"""
    print("="*50)
    print("ANOMALY DETECTION")
    print("="*50)
    
    # Setup anomaly detection
    ano_setup = setup(data=df_test)
    
    # Create isolation forest model
    iforest = create_model('iforest')
    
    # Plot results
    plot_model(iforest)
    
    return iforest

# Exploratory Data Analysis
def run_exploratory_analysis(df):
    """Run exploratory data analysis"""
    print("="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)
    
    # Basic statistics
    print("Dataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Numerical distributions
    numerical_cols = ['click_minus_sent', 'open_minus_sent', 'click_time_seconds', 
                     'sent_time_seconds', 'open_time_seconds']
    
    for col in numerical_cols:
        if col in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col], bins=50, kde=True)
            plt.title(f'Distribution of {col}')
            plt.show()
    
    # Correlation matrix
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        plt.figure(figsize=(12, 9))
        corrmat = numeric_df.corr()
        sns.heatmap(corrmat, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.show()

# Main Pipeline
def main():
    """Main analysis pipeline"""
    print("Starting Cluster Analysis Pipeline")
    print("="*50)
    
    # Setup environment
    setup_environment()
    
    # Load and prepare data
    df = load_data()
    df_balanced = create_balanced_dataset(df)
    one_hot_data = prepare_features(df_balanced)
    
    # Create test dataset
    df_unknown = df[df['final_label'].astype(str) == 'unknown']
    df_unknown_sample = df_unknown.sample(50000, random_state=1)
    df_unknown_sample['final_label'] = df_unknown_sample['final_label'].replace('unknown', 'False')
    df_true = df[df['final_label'].astype(str) == 'True']
    df_true_sample = df_true.sample(50000, random_state=1)
    df_test = pd.concat([df_true_sample, df_unknown_sample]).sample(100000, random_state=1)
    
    # Run analyses
    dt_model, rules = run_decision_tree_analysis(one_hot_data, df_balanced)
    final_model, predictions = run_pycaret_classification(df_test)
    clustering_results = run_clustering_analysis(df_test)
    embeddings = run_umap_analysis(one_hot_data)
    anomaly_model = run_anomaly_detection(df_test)
    run_exploratory_analysis(df)
    
    print("Analysis complete!")
    
    return {
        'decision_tree': (dt_model, rules),
        'classification': (final_model, predictions),
        'clustering': clustering_results,
        'umap': embeddings,
        'anomaly': anomaly_model
    }

if __name__ == "__main__":
    results = main()
