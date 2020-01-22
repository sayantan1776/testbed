import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# ------ Load and split the data ------
iris = load_iris()
iris_X = iris['data']
iris_y = pd.get_dummies(iris['target'])  # One-hot encoding using Pandas
train_X, test_X, train_Y, test_Y = train_test_split(iris_X,
                                                    iris_y,
                                                    test_size=0.3333,
                                                    random_state=3437)

# Create placeholders for passing data around the network
num_features = train_X.shape[1] # Second dimension of train_X
num_labels = train_Y.shape[1] # Second dimension of one-hot encoded labels.

X = tf.compat.v1.placeholder(tf.float32, shape=(None, num_features))
Y_correct = tf.compat.v1.placeholder(tf.float32, shape=(None, num_labels))

