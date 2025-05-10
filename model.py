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

    """
    .fit_transform(X) learns the mean, std dev, etc for each feature, and is used to learn how to transform the data. 
    It returns a horizontally stacked set of results of transformers, converting the data into a numeric format that the
    neural network can 'understand' better. Here is a simple, sample example of how we get X_processed from
    the training data:
    
    BEFORE:
    
    | Protocol_type | Service | Src_bytes | Dst_bytes |
    | -------------- | ------- | ---------- | ---------- |
    | tcp            | http    | 100        | 300        |
    | udp            | ftp     | 2000       | 100        |
    | icmp           | http    | 50         | 500        |

    AFTER
    
    | tcp | udp | icmp | http | ftp | Src_bytes_scaled | Dst_bytes_scaled |
    | --- | --- | ---- | ---- | --- | ------------------ | ------------------ |
    | 1.0 | 0.0 | 0.0  | 1.0  | 0.0 | -0.61              | 0.53               |
    | 0.0 | 1.0 | 0.0  | 0.0  | 1.0 | 1.34               | -1.22              |    # each is a row.
    | 0.0 | 0.0 | 1.0  | 1.0  | 0.0 | -0.73              | 0.69               |

    as for y_values, those represent the training data's result column, i.e., 'is_Malicious'.
    we get an array; array([0, 1, 0])

    Note that preprocessor can be used to adjust any new training data to have the same mean and mode, adapting
    it to the model's parameters.

    """

    X_processed = preprocessor.fit_transform(X)
    return X_processed, y.values, preprocessor

"""
Now we define the neural network. Like in NN Project 1, lets use a class with inbuilt definitions for the layers and
feedforward function.

Like last time, we also use torch.nn.Linear() since it is popular for simple neural networks.


"""

class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 32)  # First hidden layer: 32 neurons from the 10 input features.
        self.fc2 = nn.Linear(32, 16)  # Second hidden layer: 16 neurons generalized from 32.
        self.out = nn.Linear(16, 1)  # Output layer: 1 neuron for binary classification.

    def forward(self, x):
        x = torch.relu(self.fc1(x))   # again, applying the popular reLU  optimizer over each connection. helps with learning
        x = torch.relu(self.fc2(x))   # non-linear relationships like that of network features.  using sigmoid 'squashes' the output as a probability between 0 and 1,
        return torch.sigmoid(self.out(x))    # great for rounding for binary classification



"""
TRAINING THE MODEL

we use Binary Cross Entropy (BCE) for binary classification.

The optimizer employs the widely-used Adam algorithm to adjust the model's parameters, like its gradients
before backpropagation. The parameters() function is used by the torch.optim.Adam() function to identify 
which parameters need to be modified. 

Additionally, lr refers to the learning rate, which begins at 0.001 per pass.

we need to convert the training data (which is currently a dataframe/array) into a tensor datatype in
order to run it through the model, as tensors use float32 values so we use the torch.tensor() cfunction
to do this.

next, we use torch.tensor() with the reshaped y_train. Specifically, we have
reshaped it. for example:

BEFORE RESHAPING WITH .reshape(-1,1)

y_train = [0, 1, 0, 1, 0]

AFTER RESHAPING WITH .reshape(-1,1)

y_train_reshaped = [[0], 
                    [1], 
                    [0], 
                    [1], 
                    [0]]
                    
As before, we need to do thi because pytorch expects values to be in the form of
a 2D array, even if its just one column. reshaping it to (-1,1).
as before, -1 asks pytorch to figure out the number of rows needed based in the
original size of y_train, and 1 simply says "make it 1 column".
"""

def train_model(X_train, y_train, input_dim):
    model = Net(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

    for epoch in range(100):  # number of epochs or training cycles. More = better but takes longer.
        optimizer.zero_grad()  #for each epoch, we need to clear the gradients for next one.
        outputs = model(X_train_tensor)  # feeds the features through the model and takes the output tensor as 'outputs'
        loss = criterion(outputs, y_train_tensor)  # computes how wrong the model was in this iteration.
        loss.backward() # initiates the backpropagation. Computing how much weight each parameter has contributed to the loss so they can be adjusted.
        optimizer.step()  # adjusts parameters such that error will be reduced for the next iteration
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")  # see the loss every 10 epochs
    return model
