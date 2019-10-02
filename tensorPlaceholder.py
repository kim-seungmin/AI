import tensorflow as tf
# Create nodes in computation graph
a = tf.placeholder(tf.int32, shape=(3, 1))
b = tf.placeholder(tf.int32, shape=(1, 3))
c = tf.matmul(a, b)

# Create session object
with tf.Session() as sess:
    print(sess.run(c, feed_dict={a: [[3], [2], [1]], b: [[1, 2, 3]]}))
