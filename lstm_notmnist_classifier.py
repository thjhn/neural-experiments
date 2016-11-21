##
#  A LSTM network that learns to classify notMNIST letters.
#
#  We use the Tensor Flow library.
#
#  Author: Thomas Jahn <thomas@t-und-j.de>
##

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import random
import itertools
from six.moves import cPickle as pickle
import tensorflow as tf

# We load a pickle file which contains datasets for training,
# validation and testing.
# The file we need here is exactly what the notebook
#   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/1_notmnist.ipynb
# produces.
pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

# some constants
image_size = 28
partition_size = 2
num_labels = 10

# tweak data into a nice format
def reformat(dataset, labels):
  dataset=np.reshape(dataset,(-1,image_size/partition_size,partition_size,image_size/partition_size,partition_size))
  dataset=np.transpose(dataset,[0,1,3,2,4])
  dataset=np.reshape(dataset,(-1,image_size*image_size/partition_size/partition_size,partition_size*partition_size))
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# A Batch generator takes a dataset an produces, well ...
# batches for training and evaluation
class BatchGenerator(object):
  def __init__(self, batch_size, dataset, labels):
    self.dataset = dataset
    self.labels = labels
    self.batch_size = batch_size
    self.pointer = 0
  def next(self):
    batch_samples = []
    batch_labels = []
    for i in range(self.batch_size):
      batch_samples.append(self.dataset[self.pointer])
      batch_labels.append(self.labels[self.pointer])
      self.pointer = (self.pointer + 1)%len(self.labels)
    return batch_samples, batch_labels

# we need three batch generators
train_bg = BatchGenerator(128, train_dataset, train_labels)
valid_bg = BatchGenerator(128, valid_dataset, valid_labels)
test_bg = BatchGenerator(128, test_dataset, test_labels)

# Constants defining the model
hidden_units = 64
learning_rate = 1.0
training_steps = 5000
display_step = 100
seqlen = np.shape(train_dataset)[1]
# We log our experiemnts for tensorboard. This string in the directory
# will indicate the parameters we have used
modelstr = "hu"+str(hidden_units)+"lr"+str(learning_rate)

print("Running model "+modelstr)

#
# The Graph
#
x = tf.placeholder("float", [None, seqlen, 4], name='input')
y = tf.placeholder("float", [None, num_labels], name='labels')

# Some weights and biases
w0 = tf.Variable(tf.random_normal([4,1]), name='input_weights')
b0 = tf.Variable(tf.random_normal([1]), name='input_bias')
w = tf.Variable(tf.random_normal([hidden_units,num_labels]), name='output_weights')
b = tf.Variable(tf.random_normal([num_labels]), name='output_biases')

# Definition of the actual computation
def computeRNN(x):
  with tf.name_scope('compress_input'):
    x = tf.reshape(x, (-1,4))
    x = tf.matmul(x,w0)+b0
  with tf.name_scope('preprocess_for_rnn'):
    x = tf.reshape(x, (-1, image_size*image_size/4,1))
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, 1])
    x = tf.split(0, seqlen, x)
  lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_units)
  outputs, state = tf.nn.rnn(lstm_cell, x, dtype=tf.float32)
  with tf.name_scope('output_net'):
    outputs = tf.matmul(tf.reshape(outputs[-1:], [-1,hidden_units]),w)+b
  return outputs

rnn_computed = computeRNN(x)

with tf.name_scope('loss'):
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
      rnn_computed, y, name='LOSS'))
  tf.scalar_summary('loss', loss)
with tf.name_scope('accuracy'):  
  accuracy = tf.reduce_mean(
    tf.cast(tf.equal(
      tf.argmax(tf.nn.softmax(rnn_computed),1), tf.argmax(y,1)),"float"))
  tf.scalar_summary('accuracy', accuracy)
with tf.name_scope('train'):
  optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate).minimize(loss)

# merge all summaries
merged = tf.merge_all_summaries()

# initialize graph
init = tf.initialize_all_variables()

with tf.Session() as sess:
  # some summary writers for logging the experiments
  train_writer = tf.train.SummaryWriter('lstm-notminstclassifier-log/train-'+modelstr, sess.graph)
  valid_writer = tf.train.SummaryWriter('lstm-notminstclassifier-log/valid-'+modelstr)
  test_writer = tf.train.SummaryWriter('lstm-notminstclassifier-log/test-'+modelstr)

  # start training
  sess.run(init)
  step = 1
  while step <= training_steps:
    batch_x, batch_y = train_bg.next()
    summary, train_loss, _ , train_acc = sess.run([merged, loss, optimizer, accuracy], feed_dict={x:batch_x, y:batch_y})
    train_writer.add_summary(summary, step)
    if step % display_step == 0:
      print("At step %d on training set: loss = %2.5f, accuracy = %1.5f"%(step,train_loss,train_acc));
      batch_x, batch_y = valid_bg.next()
      summary, valid_loss , valid_acc = sess.run([merged, loss, accuracy], feed_dict={x:batch_x, y:batch_y})
      valid_writer.add_summary(summary, step)
      print("At step %d on validation set: loss = %2.5f, accuracy = %1.5f"%(step,valid_loss,valid_acc));        

    step += 1

  print("Final trainigset loss:     %2.5f" % train_loss)
  print("Final trainigset accuracy: %1.6f" % train_acc)
  batch_x, batch_y = test_bg.next()
  summary, test_loss, test_acc = sess.run(
    [merged, loss, accuracy],
    feed_dict={x:batch_x, y:batch_y})
  test_writer.add_summary(summary, step)
  print("Final test set loss:       %2.5f" % test_loss)
  print("Final test set accuracy:   %1.6f" % test_acc)
