import tensorflow as tf
import pandas as pd
import numpy as np

graph_4 = tf.Graph()
data = pd.read_csv('FuelConsumption.csv')
x_train = np.asanyarray(data[['ENGINESIZE']])
y_train = np.asanyarray(data[['CO2EMISSIONS']])

with graph_4.as_default():
    a = tf.Variable(1.0)
    b = tf.Variable(1.0)
    y = a * x_train + b

    loss = tf.reduce_mean(tf.square(y - y_train))
    opt = tf.train.GradientDescentOptimizer(0.5)
    train = opt.minimize(loss)

with tf.Session(graph=graph_4) as sess:
    sess.run(tf.global_variables_initializer())
    loss_val = []
    _, loss_, a_, b_ = sess.run([train, loss, a, b])
    loss_val.append(loss_)

print(loss_val[:-5])
