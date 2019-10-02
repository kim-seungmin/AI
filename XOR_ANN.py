import tensorflow as tf

x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [[0], [1], [1], [0]]
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W_h = tf.Variable(tf.random_normal([2, 3]))
b_h = tf.Variable(tf.random_normal([3]))
H1 = tf.sigmoid(tf.matmul(X, W_h)+b_h)
W_o = tf.Variable(tf.random_normal([3, 1]))
b_o = tf.Variable(tf.random_normal([1]))
model = tf.sigmoid(tf.matmul(H1, W_o)+b_o)
cost = -tf.reduce_mean(Y * tf.log(model) + (1 - Y) * tf.log(1 - model))
train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

predicted = tf.cast(model > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Training
    for step in range(20001):
        c, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 2000 == 0:
            print(step, c)
    # Testing
    m, p, a = sess.run([model, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print(m, p, a)

