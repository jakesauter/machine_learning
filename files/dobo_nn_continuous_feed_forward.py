import os
from decimal import Decimal, getcontext
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
### set all variables

# number of neurons in each layer
n_input = 9
n_hidden_1 = 135
n_hidden_2 = 135
n_outputs = 1

# define placeholders -- this will be my 9x1 input vector
input = tf.placeholder( tf.float32 , shape=[None , n_input])
output = tf.placeholder( tf.float32 )
### define weights and biases of the neural network (refer this article if you don't understand the terminologies)
# [!] define the number of hidden units in the first layer
# connect 2 inputs to 3 hidden units
# [!] Initialize weights with random numbers, to make the network learn
# [!] The biases are single values per hidden unit
# connect 2 inputs to every hidden unit. Add bias
# [!] The XOR problem is that the function is not linearly separable
# [!] A MLP (Multi layer perceptron) can learn to separe non linearly separable points ( you can
# think that it will learn hypercurves, not only hyperplanes)
# [!] Lets' add a new layer and change the layer 2 to output more than 1 value
# connect first hidden units to 27 hidden units in the second hidden layer
# connect the hidden units to the second hidden layer
# [!] create the new layer

#weights = []* 100	
#biaeses = [] * 100

#for i in range(n_layers)
	
weights_0 = tf.Variable(tf.truncated_normal([n_input, n_hidden_1]))
biases_0 = tf.Variable(tf.truncated_normal([n_hidden_1]))
layer_1_outputs = tf.nn.sigmoid(tf.matmul(input, weights_0) + biases_0)
weights_1 = tf.Variable(tf.truncated_normal([n_hidden_1, n_outputs]))
biases_1 = tf.Variable(tf.truncated_normal([n_hidden_2]))
layer_2_outputs = tf.nn.sigmoid(tf.matmul(input, weights_1) + biases_1)
weights_2 = tf.Variable(tf.truncated_normal([n_hidden_2, n_outputs]))
biases_2 = tf.Variable(tf.truncated_normal([n_outputs]))
#error_function = tf.nn.sigmoid_cross_entropy_with_logits(logits = [1.0], labels = output)

learning_rate = .001

logits = tf.nn.tanh(tf.matmul(layer_2_outputs, weights_2) + biases_2)

error_function = 0.5 * tf.reduce_sum(tf.subtract(logits, output) * tf.subtract(logits, output))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(error_function)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(error_function)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

execfile('get_all_entries.py')

training_inputs = x

training_outputs = y


for i in range(10000):
	_, loss = sess.run([optimizer, error_function],
		feed_dict={input: np.array(training_inputs),
			output: np.array(training_outputs)})
	print(loss)
	
error = 0
for i in range(len(x)):
	entry = x[i-1] 
	value = sess.run(logits, feed_dict={ input: np.array([entry])})
	error += ( y[i-1] - value ) * ( y[i-1] - value  ) 
	print "input: ", entry, value, '\n'

print "error: ", error

print sess.run(weights_0)
print sess.run(weights_1)

