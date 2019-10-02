from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import cv2

path_dir = './testNumber/a/'
f_list = os.listdir(path_dir)
file_list = [file for file in f_list if file.endswith(".jpg")]
file_list.sort()
test_images = input_data.read_data_sets(path_dir, one_hot=True)

X=tf.placeholder(tf.float32, [None, 784])
Y=tf.placeholder(tf.float32, [None, 10])
X_img=tf.reshape(X, [-1, 28, 28, 1])

# Convolution Layer 1
W1 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01))
CL1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding='SAME')
CL1 = tf.nn.relu(CL1)
# pooling Layer 1
PL1 = tf.nn.max_pool(CL1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# Convolution Layer 2
W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01))
CL2 = tf.nn.conv2d(PL1, W2, strides=[1,1,1,1], padding='SAME')
CL2 = tf.nn.relu(CL2)
# pooling Layer 2
PL2 = tf.nn.max_pool(CL2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# Fully Connected (FC) Layer
L_flat = tf.reshape(PL2, [-1, 7*7*64])
W3 = tf.Variable(tf.random_normal([7*7*64,10], stddev=0.01))
b3 = tf.Variable(tf.random_normal([10]))

# Model, Cost, Train
mode_LC = tf.matmul(L_flat, W3) + b3
model = tf.nn.softmax(mode_LC)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=mode_LC, labels=Y))

# Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 저장된 모델 파라미터를 가져옵니다.
    model_path = "./tmp/model.saved"
    saver = tf.train.Saver()

    saver.restore(sess, model_path)
    print("Model restored from file: %s" % model_path)

    n = len(file_list)
    images = np.zeros((n, 784))
    #prediction = np.zeros((n))

    i = 0
    for file in file_list:
        fname = path_dir + file
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        flatten = img.flatten() / 255.0
        images[i] = flatten
        image = np.reshape(images[i], (1, -1))
        prediction = sess.run(tf.argmax(model, 1), feed_dict={X: image})

        print(prediction)
        cv2.imshow(str(file), img)

        i += 1
        cv2.waitKey(0)

    #print(sess.run(tf.argmax(model, 1), feed_dict={X: images}))
cv2.destroyAllWindows()
'''
    cv2.imshow(str(file), img)
    location = (14, 14)
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    cv2.putText(img, str(prediction[i]), location, font)
'''