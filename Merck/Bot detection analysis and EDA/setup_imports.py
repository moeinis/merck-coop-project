# Setup proxy and install required packages
import os
os.environ['http_proxy'] = "http://webproxy.merck.com:8080"     
os.environ['https_proxy'] = "http://webproxy.merck.com:8080"

# Install required packages
# !pip install sqlalchemy psycopg2 evidently missingno

# Import libraries
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import psycopg2
import evidently.dashboard
import missingno

%matplotlib inline