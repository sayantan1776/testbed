import tensorflow as tf

graph_2 = tf.Graph()

with graph_2.as_default():
    v = tf.Variable([1,2,3])
    update = tf.assign(v,v+[1,1,1])

with tf.Session(graph=graph_2) as sess:
    sess.run(tf.global_variables_initializer())
    print('Initial: %s' %sess.run(v))
    for _ in range(5):
        sess.run(update)
        print(sess.run(v))
