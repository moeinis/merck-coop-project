# Clustering Analysis with PyCaret
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.clustering import *

def setup_clustering_experiment(df, ignore_features=None, session_id=123):
    """Setup PyCaret clustering experiment"""
    if ignore_features is None:
        ignore_features = ['final_label']
    
    exp_clu = setup(
        df,
        ignore_features=ignore_features,
        session_id=session_id,
        normalize=True
    )
    return exp_clu

def create_kmeans_analysis(df, num_clusters=2, ground_truth_col='final_label'):
    """Perform K-means clustering analysis"""
    print(f"Creating K-means model with {num_clusters} clusters...")
    kmeans = create_model('kmeans', num_clusters=num_clusters, ground_truth=ground_truth_col)
    
    # Plot various analyses
    plots = ['cluster', 'elbow', 'silhouette', 'distance', 'distribution']
    
    for plot_type in plots:
        try:
            print(f"Creating {plot_type} plot...")
            plot_model(kmeans, plot=plot_type)
        except Exception as e:
            print(f"Error creating {plot_type} plot: {e}")
    
    return kmeans

def create_dbscan_analysis(df):
    """Perform DBSCAN clustering analysis"""
    print("Creating DBSCAN model...")
    dbscan = create_model('dbscan')
    
    # Plot various analyses
    plots = ['cluster', 'elbow', 'silhouette', 'distance', 'distribution']
    
    for plot_type in plots:
        try:
            print(f"Creating DBSCAN {plot_type} plot...")
            plot_model(dbscan, plot=plot_type)
        except Exception as e:
            print(f"Error creating DBSCAN {plot_type} plot: {e}")
    
    return dbscan

def create_kmodes_analysis(df, num_clusters=3):
    """Perform K-modes clustering analysis"""
    print(f"Creating K-modes model with {num_clusters} clusters...")
    kmodes = create_model('kmodes', num_clusters=num_clusters)
    
    # Plot various analyses
    plots = ['cluster', 'elbow', 'silhouette', 'distance', 'distribution']
    
    for plot_type in plots:
        try:
            print(f"Creating K-modes {plot_type} plot...")
            plot_model(kmodes, plot=plot_type)
        except Exception as e:
            print(f"Error creating K-modes {plot_type} plot: {e}")
    
    return kmodes

def create_hierarchical_analysis(df, num_clusters=3):
    """Perform Hierarchical clustering analysis"""
    print(f"Creating Hierarchical clustering model with {num_clusters} clusters...")
    hclust = create_model('hclust', num_clusters=num_clusters)
    
    try:
        plot_model(hclust)
    except Exception as e:
        print(f"Error creating hierarchical cluster plot: {e}")
    
    return hclust

def analyze_cluster_results(model, ground_truth_col='final_label'):
    """Analyze clustering results against ground truth"""
    # Assign clusters
    results = assign_model(model)
    
    # Create confusion matrix
    confusion_matrix = pd.crosstab(
        results[ground_truth_col], 
        results['Cluster'], 
        rownames=['Actual'], 
        colnames=['Predicted']
    )
    
    print("Confusion Matrix:")
    print(confusion_matrix)
    
    # Plot confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, cmap="YlGnBu", cbar=False)
    plt.title("Clustering Results vs Ground Truth")
    plt.show()
    
    # Show cluster distribution
    print("\nCluster Distribution:")
    print(results['Cluster'].value_counts().sort_index())
    
    return results

def plot_feature_distribution(model, features_to_plot):
    """Plot feature distributions across clusters"""
    for feature in features_to_plot:
        try:
            print(f"Plotting distribution for {feature}...")
            plot_model(model, plot='distribution', feature=feature)
        except Exception as e:
            print(f"Error plotting {feature}: {e}")

def evaluate_clustering_performance(results_df, actual_col='final_label', cluster_col='Cluster'):
    """Evaluate clustering performance using confusion matrix metrics"""
    try:
        import pandas_ml
        from pandas_ml import ConfusionMatrix
        
        cm = ConfusionMatrix(results_df[actual_col], results_df[cluster_col])
        cm.print_stats()
        
    except ImportError:
        print("pandas_ml not available. Showing basic confusion matrix only.")
        confusion_matrix = pd.crosstab(
            results_df[actual_col], 
            results_df[cluster_col], 
            rownames=['Actual'], 
            colnames=['Predicted']
        )
        print(confusion_matrix)

def run_comprehensive_clustering_analysis(df, sample_size=200000, random_state=1):
    """Run comprehensive clustering analysis"""
    # Sample data if too large
    if len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=random_state)
        print(f"Sampled {sample_size} rows for clustering analysis")
    else:
        df_sample = df.copy()
    
    # Setup clustering experiment
    print("Setting up clustering experiment...")
    setup_clustering_experiment(df_sample)
    
    # Run different clustering algorithms
    results = {}
    
    # K-means
    print("\n" + "="*50)
    print("K-MEANS CLUSTERING")
    print("="*50)
    kmeans_model = create_kmeans_analysis(df_sample)
    kmeans_results = analyze_cluster_results(kmeans_model)
    results['kmeans'] = {'model': kmeans_model, 'results