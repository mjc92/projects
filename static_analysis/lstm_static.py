import tensorflow as tf
import numpy as np
import os
import time 
import datetime
import re
import itertools
from collections import Counter
from tensorflow.contrib import learn
from sklearn import metrics
from tensorflow.python.ops import rnn, rnn_cell

# Parameters
learning_rate = 0.01
training_iters = 100000
batch_size = 100
num_epochs = 40
display_step = 100

# Network parameters
n_input = 94
n_steps = 50 # one string is 300 chars long, 6 chars will be observed at each step
n_hidden = 128
n_classes = 2
sequence_length = 300
min_char = 33
max_char = 126

# tf Graph input
x = tf.placeholder("float",[None,sequence_length,n_input]) # takes in [batch,300,1,94]
y = tf.placeholder("float",[None,n_classes])

# Define weights
weights = {'out': tf.Variable(tf.random_normal([n_hidden,n_classes]))}
biases = {'out': tf.Variable(tf.random_normal([n_classes]))}

"""
sequence_length=300
num_classes=2

vocab_size=100 # default
num_filters=256
l2_reg_lambda=0.0

batch_size=100
num_epochs=150
evaluate_every=1000
checkpoint_every=1000
"""
allow_soft_placement=True
log_device_placement=True

# File directory
#train_FILE = 'train.txt'
#test_FILE = 'test.txt'
train_FILE = 'train.txt'
test_FILE = 'test.txt'

# Load data
print("Loading data...")
# import training and test data

xy_train = np.loadtxt(train_FILE,unpack=True,dtype='int')
xy_test = np.loadtxt(test_FILE,unpack=True,dtype='int')
print("Data loaded!")

xy_train = xy_train.T
xy_test = xy_test.T
print "Training data shape: "+str(xy_train.shape)
print "Test data shape: "+str(xy_test.shape)

# get training and test sets
x_train = xy_train[:,0:300]
y_train = xy_train[:,300]
x_test = xy_test[:,0:300]
y_test = xy_test[:,300]

# use portion of data
x_train = x_train[0:50000]
y_train = y_train[0:50000]

t_idx = np.random.permutation(len(y_test))
t_idx=t_idx[0:10000]
x_test = x_test[t_idx,:]
y_test = y_test[t_idx]

# change x to one-hot
def embed_x(x,max_char,min_char):
    m,n = x.shape
    out = np.zeros((m*n,max_char-min_char+1)) # (batch*length,94)
    x_reshape = np.reshape(x,(m*n)) # (batch_length,1)
    out[xrange(m*n),x_reshape-min_char]=1
    out=np.reshape(out,(m,n,-1))
    return out

# change y to 2 classes
def embed_y(y):
    y = np.concatenate([1-y,y],axis=1)
    return y

def RNN(x, weights, biases):
    with tf.device('/gpu:0'):
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Permuting batch_size and n_steps
        print 'Permuting batch size and number of steps...'
        x = tf.transpose(x, [1, 0, 2])
        # Reshaping to (n_steps*batch_size, n_input)
        print 'Reshaping to (n_steps * batch_size, n_input)...'
        x = tf.reshape(x, [-1, n_input])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        print 'Splitting to get a list of n_step tensors of shape (batch_size, n_input)...'
        x = tf.split(0, sequence_length, x)

        # Define a lstm cell with tensorflow
        lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0,state_is_tuple=True)

        # Get lstm cell output
        outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()
# create saver
#saver = tf.train.Saver()
# Launch the graph
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
with tf.Session(config=tf.ConfigProto(log_device_placement=True,gpu_options=gpu_options)) as sess:
    sess.run(init)
    # Keep training until reach max iterations
    num_batches=len(y_train)/batch_size
    steps=1
    for i in xrange(num_epochs):
        # shuffle data before dividing into batches
        tr_idx=np.random.permutation(len(y_train))
        x_train=x_train[tr_idx,:]
        y_train=y_train[tr_idx]

        for j in xrange(num_batches):
            batch_x = x_train[j*batch_size:(j+1)*batch_size]
            batch_y = y_train[j*batch_size:(j+1)*batch_size]
            # embed x and y
            batch_x = embed_x(batch_x,max_char,min_char)
            batch_y = embed_y(np.reshape(batch_y,(batch_y.shape[0],1)))
    #        batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Reshape data to get 28 seq of 28 elements
    #        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
            # batch_x: [100, 300, 94]
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            if (steps) % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            	print "Iter " + str(steps) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)
            steps+=1
        print "End of %dth epoch............\n"%(i+1)
    print "Optimization Finished! Saving variables..."
    #save_path = saver.save(sess,"/saves/"+"model_"+datetime.datetime.now().isoformat()+".ckpt")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    #test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    #test_label = mnist.test.labels[:test_len]
    x_test = embed_x(x_test, max_char,min_char)
    y_test = embed_y(np.reshape(y_test,(y_test.shape[0],1)))

    print "Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: x_test, y: y_test})
#        sess.run(accuracy, feed_dict={x: test_data, y: test_label})
