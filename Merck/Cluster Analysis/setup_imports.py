# Setup and Imports for Cluster Analysis
# Environment Setup and Required Libraries

import os
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
import psycopg2

# Plotting libraries
import plotly.offline as pyo
import plotly.graph_objs as go
from pycaret.utils import enable_colab

# Machine Learning libraries
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn import metrics, tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.decomposition import PCA
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

# Clustering and dimensionality reduction
import umap
import hdbscan

# Visualization
from yellowbrick.model_selection import FeatureImportances
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
import graphviz

# Setup environment
def setup_environment():
    """Configure environment settings and display options"""
    # Set proxy if needed (comment out if not required)
    # os.environ['http_proxy'] = "http://webproxy.merck.com:8080"     
    # os.environ['https_proxy'] = "http://webproxy.merck.com:8080"
    
    # Configure pandas display options
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    
    # Configure plotting
    plt.style.use('default')
    sns.set(style='white', rc={'figure.figsize': (10, 8)})
    pyo.init_notebook_mode()
    enable_colab()

if __name__ == "__main__":
    setup_environment()
    print("Environment setup complete!")
