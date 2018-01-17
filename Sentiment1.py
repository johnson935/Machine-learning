import tensorflow as tf
import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data
''' 
input > weight > hidden layer 1 (activation function) > weights > hidden layer 2 
(activation function) > weights > output layer. This is feed forward

compare output to intended output > cost / loss function (cross entropy)
Optimization function (optimizer) > minimize cost (AdamOptimizer , SGD, AdaGrad)

Backpropagation 

Feed forward + backprop = epoch

'''
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
from Neuralnetwork2 import create_feature_sets_and_labels

train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt','neg.txt')

'''
0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0]
'''
# hidden layer nodes
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2
# batches of 100 to manipulated the weights
batch_size = 100

# height * width
x = tf.placeholder('float', [None, len(train_x[0])])  # sets the shape of the array input
y = tf.placeholder('float') #labels for the input data in order to classify them into the 10 classes


def neural_network_model(data):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}
    # (input data * weights) +biases

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
# input data now becomes the output data of layer1
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    # input data now becomes output of layer 2
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
#outputs a probability
    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
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
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                
                
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size

            print('epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))


train_neural_network(x)
