import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pymysql
import mysql.connector
import numpy as np

""" Retrieving the data from the mysql database. For some reason, using pymysql resolved the sha_password_authentication
    error that using mysql.connector would yield. """

def load_data():
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='password',
        database='nn_project_2_database'
    )

    """
    predefined query to fetch all the columns of training data - we need to use oit to train the mo del
    """

    query = "SELECT Protocol_type, Service, Flag, Src_bytes, Dst_bytes, Count, Serror_rate, Rerror_rate, Malicious FROM connection_data"
    df = pd.read_sql(query, conn)
    conn.close()
    return df.dropna()  # we can use dropna() function to drop rows with null values to clean the data.


def preprocess_data(df):
    categorical_cols = ['Protocol_type', 'Service', 'Flag']
    numerical_cols = ['Src_bytes', 'Dst_bytes', 'Count', 'Serror_rate', 'Rerror_rate']
    target_col = 'Malicious'

    """
    Though not super necessary, it is more organized to split the data into categorical and numerical columns. it makes
    no difference to the training as X simply adds them in a dataframe, from df (which is the full dataframe).
    
    y is simply the 0 or 1: is it malicious? used for training (single col)
    """

    X = df[categorical_cols + numerical_cols]
    y = df[target_col].astype(int)

    """
    Here, the data is processed into a form that the neural network can be trained on,
    similar to how Tokenization is done in large langauge models.
    
    For example, the protocols ['TCP', 'UDP', 'ICMP'] may become [1, 0, 0], [0, 1, 0] for differentiation.
    so that these binary representations can be used to train the neural network.
    
    Note that 'cat' stands for 'categorical' and 'num' represents the numerical data.
    we use the StandardScaler function to normalize them:

    It transforms features to have mean = 0 and std = 1, and helps in the (nn's) learning process
    """

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', StandardScaler(), numerical_cols)
        ]
    )
