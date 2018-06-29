import load_data
import tensorflow as tf 

number_inputs = 9
number_outputs = 1
learning_rate = 0.001
training_epochs = 100
layer_1_nodes = 50
layer_2_nodes = 100
layer_3_nodes = 50

with tf.variable_scope('input'):
	X = tf.placeholder(tf.float32, name='input', shape=(None, number_inputs))

with tf.variable_scope('layer_1'):
	weights = tf.get_variable(name='weights1', shape=[number_inputs, layer_1_nodes], initializer=tf.contrib.layers.xavier_initializer())
	biases = tf.get_variable(name='biases1', shape = [layer_1_nodes], initializer=tf.zeros_initializer())
	layer_1_output = tf.nn.relu(tf.matmul(X,weights) + biases)

with tf.variable_scope('layer_2'):
	weights = tf.get_variable(name='weights2', shape=[layer_1_nodes, layer_2_nodes], initializer=tf.contrib.layers.xavier_initializer())
	biases = tf.get_variable(name='biases2', shape=[layer_2_nodes], initializer=tf.zeros_initializer())
	layer_2_output = tf.nn.relu(tf.matmul(layer_1_output,weights)+biases)

with tf.variable_scope('layer_3'):
	weights = tf.get_variable(name='weights3', shape=[layer_2_nodes, layer_3_nodes], initializer=tf.contrib.layers.xavier_initializer())
	biases = tf.get_variable(name='biases3', shape=[layer_3_nodes], initializer=tf.zeros_initializer())
	layer_3_output = tf.nn.relu(tf.matmul(layer_2_output,weights)+biases)

with tf.variable_scope('output'):
	weights = tf.get_variable(name='weights4', shape=[layer_3_nodes, number_outputs], initializer=tf.contrib.layers.xavier_initializer())
	biases = tf.get_variable(name='biases4', shape=[number_outputs], initializer=tf.zeros_initializer())
	prediction = tf.nn.relu(tf.matmul(layer_3_output,weights)+biases)

with tf.variable_scope('cost'):
	Y = tf.placeholder(tf.float32, shape=(None,1))
	cost = tf.reduce_mean(tf.squared_difference(prediction, Y))

with tf.variable_scope('train'):
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.variable_scope('logging'):
	tf.summary.scalar('current_cost',cost)
	summary = tf.summary.merge_all()

saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	training_writer = tf.summary.FileWriter('log/training', sess.graph)
	testing_writer = tf.summary.FileWriter('log/testing', sess.graph)
	for epochs in range(training_epochs):
		sess.run(optimizer, feed_dict={X: load_data.X_scaled_training, Y: load_data.Y_scaled_training})
		if epochs%5==0:
			training_cost, training_summary = sess.run([cost, summary], feed_dict={X: load_data.X_scaled_training, Y: load_data.Y_scaled_training})
			testing_cost, testing_summary = sess.run([cost, summary], feed_dict={X: load_data.X_scaled_testing, Y: load_data.Y_scaled_testing})
			print(epochs,training_cost, testing_cost)
			training_writer.add_summary(training_summary,epochs)
			testing_writer.add_summary(testing_summary, epochs)
	print("Training Complete")
	final_training_cost = sess.run(cost, feed_dict={X: load_data.X_scaled_training, Y: load_data.Y_scaled_training})
	final_testing_cost = sess.run(cost, feed_dict={X: load_data.X_scaled_testing, Y: load_data.Y_scaled_testing})
	print("Final training cost {}".format(final_training_cost))
	print("Final testing cost {}".format(final_testing_cost))
	saver.save(sess, "log/model_1.ckpt")