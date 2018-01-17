import sys
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages')

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn


''' 
input > weight > hidden layer 1 (activation function) > weights > hidden layer 2 
(activation function) > weights > output layer. This is feed forward

compare output to intended output > cost / loss function (cross entropy)
Optimization function (optimizer) > minimize cost (AdamOptimizer , SGD, AdaGrad)

Backpropagation 

Feed forward + backprop = epoch

'''
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# 10 classes, 0-9
'''
0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0]
'''

hm_epochs = 3
n_classes = 10
# batches of 128 to manipulated the weights
batch_size = 128


chunk_size = 28
n_chunks = 28
rnn_size = 128


# height * width
x = tf.placeholder('float', [None, n_chunks, chunk_size])  # sets the shape of the array input
y = tf.placeholder('float') #labels for the input data in order to classify them into the 10 classes


def recurrent_neural_network(x):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}
    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    b = int(0)
    x = tf.split(x, n_chunks, 0)
    
    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    
    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']
#outputs a probability
    return output

def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    #labels: each row label must be a valid probability distribution y are the labels
    #logits: unscaled log probability meant to be the hypothesis i.e. prediction
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    print(cost)
    print(y)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)



# starting session
    with tf.Session() as sess:

        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            # the _ in range means a variable that we do not require
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images.reshape((-1, n_chunks, chunk_size)), y: mnist.test.labels}))


train_neural_network(x)
