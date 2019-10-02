import tensorflow as tf
# Create nodes in computation graph
node1 = tf.constant(3, dtype=tf.int32)
node2 = tf.constant(5, dtype=tf.int32)
node3 = tf.add(node1, node2)

# Create session object
with tf.Session() as sess:
    print('node1 + node2 = ', sess.run(node3))



