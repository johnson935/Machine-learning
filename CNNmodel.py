import sys
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages')

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
''' 
input > weight > hidden layer 1 (activation function) > weights > hidden layer 2 
(activation function) > weights > output layer. This is feed forward

compare output to intended output > cost / loss function (cross entropy)
Optimization function (optimizer) > minimize cost (AdamOptimizer , SGD, AdaGrad)

Backpropagation 

Feed forward + backprop = epoch

'''
#dropout : not all neuron work at the same time
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# 10 classes, 0-9
'''
0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0]
'''


n_classes = 10
# batches of 100 to manipulated the weights
batch_size = 128

# height * width
x = tf.placeholder('float', [None, 784])  # sets the shape of the array input
y = tf.placeholder('float') #labels for the input data in order to classify them into the 10 classes

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

# moving convolution 1x1
def conv2d(x, W):
    #                           moving convolution
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# moving pooling 2x2
def maxpool2d(x):
    #                           size of window      moving of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_neural_network(x):
    #5x5 convolution take 1 input and product 32 outputs
    weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
               #5 x5 convolution with 32 inputs from W_conv1
               'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
               #7 x 7 convloution with 64 inputs
               'W_fc': tf.Variable(tf.random_normal([7*7*64, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))
               }

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes]))
              }

    x = tf.reshape(x, shape=[-1, 28, 28, 1])
   # convolution = convolution*pooling
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)
    #fc = full convlution
    fc = tf.reshape(conv2, [-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])

    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']



    return output


def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    #labels: each row label must be a valid probability distribution y are the labels
    #logits: unscaled log probability meant to be the hypothesis i.e. prediction
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    print(cost)
    print(y)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

    hm_epochs = 10

# starting session
    with tf.Session() as sess:

        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            # the _ in range means a variable that we do not require
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


train_neural_network(x)
