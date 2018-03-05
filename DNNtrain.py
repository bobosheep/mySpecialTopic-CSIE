import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn.model_selection as sk

def add_layer(inputs, insize, outsize, activation_function=None):
    Weights = tf.Variable(tf.random_normal([insize, outsize]))
    biases = tf.Variable(tf.zeros([1, outsize]))
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    
    if activation_function is None:
        output = Wx_plus_b
    else:
        output = activation_function(Wx_plus_b)
    return output

def correct_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys})
    return result

hawaF = './data/features-hawa.csv'
hawaL = './data/onehotLabels-hawa.csv'
myF = './data/featuredata-corr.csv'
myL = './data/onehotdata.csv'

features = pd.read_csv(myF, header = None)
features = np.array(features)
labels = np.loadtxt(myL, delimiter = ',')
hawafeatures = np.loadtxt(hawaF, delimiter = ',')
labels = np.loadtxt(hawaL, delimiter=',')
features = np.hstack([features, hawafeatures])
#print(features)
#print(hawafeatures)
train_x, test_x, train_y, test_y = sk.train_test_split(features, labels, test_size = 0.1)

train_x = np.array(train_x, dtype=np.float32)
train_y = np.array(train_y, dtype=np.float32)
test_x = np.array(test_x, dtype=np.float32)
test_y = np.array(test_y, dtype=np.float32)

# input
xs = tf.placeholder(tf.float32, [None, 3])
# output
ys = tf.placeholder(tf.float32, [None, 9])


# add hidden layer
h1 = add_layer(xs, 3, 18, activation_function = tf.nn.relu)
#h2 = add_layer(h1, 18, 14, activation_function = tf.nn.relu)

# add output layer
prediction = add_layer(h1, 18, 9, activation_function = tf.nn.softmax)


loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices = [1]))

train_step = tf.train.GradientDescentOptimizer(0.3).minimize(loss)
# initialization
init = tf.global_variables_initializer()

# run session
sess = tf.Session()
sess.run(init)
for i in range(1500):
    # training
    sess.run(train_step, feed_dict={xs: train_x, ys: train_y})
    if(i % 100 == 0):
        # to see the step improvement
        print('%d epoch accuracy: %f' % (i, correct_accuracy(test_x, test_y)))
