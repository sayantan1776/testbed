import tensorflow as tf

graph_3 = tf.Graph()

with graph_3.as_default():
    ph_ = tf.placeholder(tf.float32)
    var_ = tf.Variable([1.0,1.0,1.0])
    add_ = ph_ + var_

with tf.Session(graph=graph_3) as sess:
    sess.run(tf.global_variables_initializer())
    res_ = sess.run(add_, feed_dict={ph_: [1.0,2.0,3.0]})
    print(res_)
