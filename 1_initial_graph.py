import tensorflow as tf

graph_1 = tf.Graph()

with graph_1.as_default():
    a = tf.constant([2,3,4], name='const_a')
    b = tf.constant([4,5,6], name='const_b')
    c = tf.add(a,b)

with tf.Session(graph=graph_1) as sess:
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))
