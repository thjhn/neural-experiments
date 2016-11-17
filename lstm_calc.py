##
#  A LSTM network that learns to evaluate simple computations
#  in RPN.
#
#  We use the Tensor Flow library.
#
#  Author: Thomas Jahn <thomas@t-und-j.de>
##

from __future__ import print_function
import tensorflow as tf
import numpy as np
import random

##
#  First we generate sequences.
#
#  The operations we allow in our RPNs are +, -, * and /.
#  The only numbers allowed for input are digits 0-9.
#  This gives us a vocabulary of 14 elements.
#
#  Examples:
#    12+ is 3
#    13*55+* is 30
#
#  We generate valid RPNs using a random binary tree of
#  maximal depth max_tree_depth. This produces sequences
#  of maximal length 2**(max_tree_depth+1)-1.
#  Each node is an CalcNode object. Upon initialisation it
#  is randomly asigned to be a digit (i.e. a leaf) or an
#  operation.
#
#  The output is a sequence of digits, possibly preceded by
#  a minus sign. The maximal length a sequence will have is
#  max_seq_length_out.
#  A sequence that is shorter will be padded by +.
##

max_tree_depth = 3
max_seq_length_in = 2**(max_tree_depth+1)-1
max_seq_length_out = 5
vocabulary_size = 14


class CalcNode(object):
    def __init__(self, depth):
        if depth>=max_tree_depth:
            # we are forced to use a digit as we are in the
            # deepest level allowed.
            self.value = random.randrange(10)
        else:
            # Dice whether to fill in a operation (prop=3/4)
            # or a digit.
            if random.randrange(4)<3:
                # it's an op
                self.value = random.randrange(10,14)
                self.left = CalcNode(depth+1)
                self.right = CalcNode(depth+1)
            else:
                #it's a digit
                self.value = random.randrange(10)

    # produce a list representation of the RPN
    def read_list(self):
        if self.value < 10:
            return [self.value]
        else:
            return self.left.read_list()+self.right.read_list()+[self.value]

    # compute the value of the string.
    # since the RPN was generated randomly it my happen that a division
    # by zero occurs. Then an exception will be thrown. We need this exception
    # later, so do not catch it!
    def calc(self):
        if self.value < 10:
            return str(self.value)
        else:
            a = int(self.left.calc())
            b = int(self.right.calc())
            if self.value == 10:
                return a+b
            elif self.value == 11:
                return a-b
            elif self.value == 12:
                return a*b
            elif self.value == 13:
                return a/b

# convert a list of int representation into a string
def rpn_list_to_string(rpn):
    ret = ""
    for item in rpn:
        if item < 10:
            ret += str(item)
        elif item == 10:
            ret += "+"
        elif item == 11:
            ret += "-"
        elif item == 12:
            ret += "*"
        elif item == 13:
            ret += "/"
    return ret

# Objects of SampleGenerator actually produce input for the RPN.
class SampleGenerator(object):
    def __init__(self, n_samples=1000):
        self.samples = []
        self.pointer = 0
        while len(self.samples) < n_samples:
            s_calc = CalcNode(0)
            try:
                # we try to compute the result. if it fails (division by zero)
                # we ignore this sample and generate a new one.
                s_value = s_calc.calc()
                self.samples.append([s_calc.read_list(),s_value])                
                samples.append(c)
            except:
                pass
            
    def next(self, batch_size):
        batch_values = []
        batch_labels = []
        batch_vsizes = []
        for _ in range(batch_size):
            val = np.zeros((max_seq_length_in, vocabulary_size),dtype=float)
            list_val = self.samples[self.pointer][0]
            for i in range(len(list_val)):
                val[i, list_val[i]] = 1.0 
            batch_values.append(val)

            label = np.zeros((max_seq_length_out, vocabulary_size),dtype=float)
            str_label = str(self.samples[self.pointer][1])
            for i in range(len(str_label), max_seq_length_out):
                str_label += "+"
            for i in range(len(str_label[-max_seq_length_out:])):
                # only + (for padding), - and digits will occur
                if str_label[i] == '+':
                    label[i, 10] = 1.0
                elif str_label[i] == '-':
                    label[i, 11] = 1.0
                else:
                    label[i, int(str_label[i])] = 1.0
            batch_labels.append(label)
            batch_vsizes.append(len(self.samples[self.pointer][0]))
            self.pointer = (self.pointer+1)%len(self.samples)
        return batch_values, batch_labels, batch_vsizes

    
##
#  The Model
##

# Parameters
learning_rate = 1.0
#training_steps = 100000
training_steps = 3
batch_size = 128
display_step = 100

# Network Parameters
hidden_cells = 64

# Create training and test sets
training_set = SampleGenerator(n_samples=training_steps*batch_size)
test_set = SampleGenerator(n_samples=batch_size)

# Let's build a Graph
# Inputs. x the input sequence, y the output seqence, seqlen the
# length of each sequence in x
x = tf.placeholder("float", [None, max_seq_length_in, vocabulary_size])
y = tf.placeholder("float", [None, max_seq_length_out, vocabulary_size])
seqlen = tf.placeholder(tf.int32, [None])

# Weight and bias for the output
w = tf.Variable(tf.random_normal([hidden_cells, vocabulary_size]))
b = tf.Variable(tf.random_normal([vocabulary_size]))

def dynamicRNN(x, seqlen, weights, biases):
    # We squeeze the data such that it fits into our RNN
    x = tf.split(0, max_seq_length_in,
                 tf.reshape(tf.transpose(x, [1, 0, 2]),
                 [-1, vocabulary_size]))

    # Now we are ready to put the data into the 'in' stage
    # of our network of LSTM cells.
    with tf.variable_scope('in'):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_cells)
        _, state = tf.nn.rnn(lstm_cell, x, dtype=tf.float32,
                                    sequence_length=seqlen)

    # For the 'out' network we do not have any input. So let's
    # create noise!
    noisy_input = tf.zeros((max_seq_length_out*batch_size,hidden_cells),dtype=tf.float32); # just zeros for now
    noisy_input = tf.split(0, max_seq_length_out, noisy_input)

    # And we are ready for the 'out' stage
    with tf.variable_scope('out'):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_cells)
        outputs, _ = tf.nn.rnn(lstm_cell, noisy_input,
                               initial_state=state, dtype=tf.float32)

    # we have to apply a linear transformation for each output character
    result = []
    for i in range(max_seq_length_out):
        result.append(tf.nn.softmax(tf.matmul(outputs[i], weights)+biases))
    # squeeze back
    result = tf.transpose(result, [1, 0, 2])
    return result

prediction = dynamicRNN(x,seqlen, w, b)
first_choice = tf.argmax(prediction,2)
loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(-tf.log(prediction)*y,2),1))
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction,2), tf.argmax(y,2)),"float"))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step <= training_steps:
        # get batch and train
        batch_x, batch_y, batch_seqlen = training_set.next(batch_size)
        training_loss, _, training_accuracy = sess.run(
            [loss, optimizer, accuracy],
            feed_dict={x:batch_x, y:batch_y, seqlen:batch_seqlen})
        
        # sometimes write something useful
        if step % display_step == 0:
            print("At step %d: trainig set loss = %2.5f, training set accuracy = %1.5f"
                  % (step, training_loss, training_accuracy))

        step += 1
        
    print("Final trainigset loss:     %2.5f" % training_loss)
    print("Final trainigset accuracy: %1.6f" % training_accuracy)
        
    batch_x, batch_y, batch_seqlen = test_set.next(batch_size)
    test_loss, _, test_accuracy, test_1st_choice = sess.run(
        [loss, optimizer, accuracy, first_choice],
        feed_dict={x:batch_x, y:batch_y, seqlen:batch_seqlen})
    print("Final test set loss:       %2.5f" % test_loss)
    print("Final test set accuracy:   %1.6f" % test_accuracy)

    print("Some sample predictions:")
    for i in range(50):
        input = rpn_list_to_string(
            np.argmax(batch_x[i],1)[:batch_seqlen[i]]
        )
        label = rpn_list_to_string(np.argmax(batch_y[i],1))
        predict = rpn_list_to_string(test_1st_choice[i])
        print(" * Prediction for %31s is %5s, correct is %5s."%(input,predict,label))
