import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

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

# Create weights and bias variables
weights = tf.Variable(tf.random.normal((num_features, num_labels)))
biases = tf.Variable(tf.random.normal((num_labels,)))

# Operations for logistic regression
# 1. Multiply X by weights
# 2. Add bias to product
# 3. Apply sigmoid function to sum
mult_op = tf.matmul(X, weights)
add_op = tf.add(mult_op, biases)
sigmoid_op = tf.sigmoid(add_op)

# Setting up learning duration (no. of epochs) and learning rate
num_epochs = 5000
learningRate = tf.compat.v1.train.exponential_decay(learning_rate=0.001,
                                          global_step=0,
                                          decay_steps=100,
                                          decay_rate=0.95,
                                          staircase=True)
# Setting up loss function
loss_val = tf.nn.l2_loss(sigmoid_op - Y_correct)

# Setting up a single optimization step
#  - reducing the L2 norm between result from sigmoid operation
#    and correct label (one-hot encoded), using an exponentially 
#    decaying learning rate.
opt_out = tf.compat.v1.train.GradientDescentOptimizer(learningRate).minimize(loss_val)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    # argmax(activation_OP, 1) returns the label with the most probability
    # argmax(yGold, 1) is the correct label
    correct_predictions_OP = tf.equal(tf.argmax(input=sigmoid_op,axis=1),tf.argmax(input=Y_correct,axis=1))

    # If every false prediction is 0 and every true prediction is 1, the average returns us the accuracy
    accuracy_OP = tf.reduce_mean(input_tensor=tf.cast(correct_predictions_OP, "float"))

    # Summary op for regression output
    activation_summary_OP = tf.compat.v1.summary.histogram("output", sigmoid_op)

    # Summary op for accuracy
    accuracy_summary_OP = tf.compat.v1.summary.scalar("accuracy", accuracy_OP)

    # Summary op for cost
    cost_summary_OP = tf.compat.v1.summary.scalar("cost", loss_val)

    # Summary ops to check how variables (W, b) are updating after each iteration
    weightSummary = tf.compat.v1.summary.histogram("weights", weights.eval(session=sess))
    biasSummary = tf.compat.v1.summary.histogram("biases", biases.eval(session=sess))

    # Merge all summaries
    merged = tf.compat.v1.summary.merge([activation_summary_OP, accuracy_summary_OP, cost_summary_OP, weightSummary, biasSummary])

    # Summary writer
    writer = tf.compat.v1.summary.FileWriter("summary_logs", sess.graph)
    cost = 0
    diff = 1
    epoch_values = []
    accuracy_values = []
    cost_values = []
    for ep in range(num_epochs):
        # Run training step
        step = sess.run(opt_out, feed_dict={X: train_X, Y_correct: train_Y})
        # Report occasional stats
        if ep % 100 == 0:
            # Add epoch to epoch_values
            epoch_values.append(ep)
            # Generate accuracy stats on test data
            train_accuracy, newCost = sess.run([accuracy_OP, loss_val], feed_dict={X: train_X, Y_correct: train_Y})
            # Add accuracy to live graphing variable
            accuracy_values.append(train_accuracy)
            # Add cost to live graphing variable
            cost_values.append(newCost)
            # Re-assign values for variables
            diff = abs(newCost - cost)
            cost = newCost

            #generate print statements
            print("Step %d, training accuracy %g, cost %g, change in cost %g"%(ep, train_accuracy, newCost, diff))
    # How well do we perform on held-out test data?
    print("Final accuracy on test set: %s" %
          str(sess.run(accuracy_OP, feed_dict={X: test_X, Y_correct: test_Y})))

fig, ax1 = plt.subplots()
ax1.plot(cost_values, label='Cost function', color='red')
plt.legend()
ax2 = ax1.twinx()
ax2.plot(accuracy_values, label='Accuracy', color='blue')
# plt.plot(cost_values, label='Cost function')
# plt.plot(accuracy_values, label='Accuracy')
plt.legend()
plt.show()
