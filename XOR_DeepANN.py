import tensorflow as tf

x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [[0], [1], [1], [0]]
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W1 = tf.Variable(tf.random_normal([2, 10]))
b1 = tf.Variable(tf.random_normal([10]))
H1 = tf.sigmoid(tf.matmul(X, W1)+b1)

W2 = tf.Variable(tf.random_normal([10, 10]))
b2 = tf.Variable(tf.random_normal([10]))
H2 = tf.sigmoid(tf.matmul(H1, W2)+b2)

W3 = tf.Variable(tf.random_normal([10, 10]))
b3 = tf.Variable(tf.random_normal([10]))
H3 = tf.sigmoid(tf.matmul(H2, W3)+b3)

W4 = tf.Variable(tf.random_normal([10, 1]))
b4 = tf.Variable(tf.random_normal([1]))
model = tf.sigmoid(tf.matmul(H3, W4)+b4)

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


tf.nn.softmax_cross_entropy_with_logits()