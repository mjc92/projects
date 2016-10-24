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

saver_name="model"+str(time.time())

# Parameters
sequence_length=300
num_classes=2

vocab_size=100 # default
filter_sizes=[3,4,5,6]
num_filters=256
dropout_keep_prob=0.5
l2_reg_lambda=0.0

batch_size=100
num_epochs=30
evaluate_every=100
checkpoint_every=1000

allow_soft_placement=True
log_device_placement=True

# File directory
train_FILE = 'train.txt'
test_FILE = 'test.txt'
#train_FILE = 'skFIFS_train5.txt'
#test_FILE = 'skFIFS_test5.txt'

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

x_train, x_dev = x_train[:-1000], x_train[-1000:]
y_train, y_dev = y_train[:-1000], y_train[-1000:]
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

# optional - use portion of data
tr_idx = np.random.permutation(len(y_train))
t_idx = np.random.permutation(len(y_test))
x_train = x_train[tr_idx[0:50000],:]
y_train = y_train[tr_idx[0:50000]]
x_test = x_test[t_idx[0:10000],:]
y_test = y_test[t_idx[0:10000]]

max_char = np.amax(x_train)
min_char = np.amin(x_train)
vocab_size = max_char-min_char+1
print "Largest char is: %d -> "%max_char+str(chr(max_char))
print "Smallest char is: %d -> "%min_char+str(chr(min_char))
print "Total number of characters: " + str(vocab_size)

# change x to one-hot
def embed_x(x,max_char,min_char):
    m,n = x.shape
    out = np.zeros((m*n,max_char-min_char+1)) # (batch*length,94)
    x_reshape = np.reshape(x,(m*n)) # (batch_length,1)
    out[xrange(m*n),x_reshape-min_char]=1
    out=np.reshape(out,(m,n,1,-1))
    return out
x_dev = embed_x(x_dev,max_char,min_char)

# change y to 2 classes
def embed_y(y):
    y = np.concatenate([1-y,y],axis=1)
    return y
y_train = embed_y(np.reshape(y_train,(y_train.shape[0],1)))
y_dev = embed_y(np.reshape(y_dev,(y_dev.shape[0],1)))

class TextCNN(object):
    def __init__(self,sequence_length,num_classes,vocab_size,filter_sizes,num_filters,l2_reg_lambda=0.0):
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length,1,vocab_size],name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes],name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32,name="dropout_keep_prob")
        
        # keep track of l2 regularization loss
        l2_loss = tf.constant(0.0)
        
        # no need for an embedding layer
        
        # create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        with tf.device('/gpu:0'):
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv_maxpool-%s" % filter_size):
                    # convolution layer
                    # input tensor = [batch, in_height, in_width, in_channels]
                    #              = [100,   300,       1,        94]
                    # filter shape = [filter_height, filter_width, in_channels, out_channels]
                    filter_shape = [filter_size,1,vocab_size,num_filters]
                    
                    W = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]),name="b")
                    conv = tf.nn.conv2d(self.input_x,W,strides=[1,1,1,1],padding="VALID",name="conv")
                    # nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv,b),name="relu")
                    # maxpooling over outputs
                    pooled = tf.nn.max_pool(h,ksize=[1,sequence_length-filter_size+1,1,1],strides=[1,1,1,1],
                                           padding="VALID",name="pool")
                    pooled_outputs.append(pooled)
                    
            # combine all pooled features
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool = tf.concat(3, pooled_outputs)
            self.h_pool_flat = tf.reshape(self.h_pool,[-1,num_filters_total])

            # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.get_variable("W", shape=[num_filters_total, num_classes],
                                   initializer = tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.h_drop, W,b,name="scores")
                self.predictions = tf.argmax(self.scores, 1, name="predictions")

            # Calculate mean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y,1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")            

# Training
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5,allow_growth=True)
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      gpu_options=gpu_options,  
      allow_soft_placement=allow_soft_placement,
      log_device_placement=log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=2,
            vocab_size=vocab_size,
            filter_sizes=filter_sizes,
            num_filters=num_filters,
            l2_reg_lambda=l2_reg_lambda)
        
        # Define training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars,global_step=global_step)
        
        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cnn.loss)
        acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())
        
        # Write vocabulary
#        vocab_processor.save(os.path.join(out_dir,"vocab"))
        
        # Initialize all variables
        sess.run(tf.initialize_all_variables())
        def train_step(x_batch, y_batch,i):
            # A single training step
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: dropout_keep_prob
            }
            _, step,summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            if i%10==9:
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)
            
        def dev_step(x_batch, y_batch, writer=None):
            # Evaluates model on a dev?test? set
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)
    
        # Training loop. For each batch...
        for i in xrange(num_epochs):
            for j in xrange(int(x_train.shape[0]/batch_size)):
                x_batch=embed_x(x_train[j*batch_size:(j+1)*batch_size-1,:],max_char,min_char)
                y_batch=y_train[j*batch_size:(j+1)*batch_size-1]
                train_step(x_batch, y_batch,j)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=allow_soft_placement,
      log_device_placement=log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Collect the predictions here
        all_predictions = []

        for i in xrange(int(x_test.shape[0]/batch_size)):
            x_test_batch=embed_x(x_test[i*batch_size:(i+1)*batch_size-1,:],max_char,min_char)
            y_test_batch=y_test[i*batch_size:(i+1)*batch_size-1]
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

if len(y_test)>len(all_predictions):
    y_test=y_test[0:len(all_predictions)]

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
    fpr, tpr, thresholds = metrics.roc_curve(y_test,all_predictions,pos_label=1)
    print "AUC: %f" % metrics.auc(fpr,tpr)
# save values
np.savetxt("all_predictions.txt",all_predictions,fmt="%d",delimiter=' ')
np.savetxt("actual_classes.txt",y_test,fmt="%d",delimiter=' ')