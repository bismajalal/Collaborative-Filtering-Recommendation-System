import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from jproperties import Properties
import PreProcessing

epochs = 20
num_batches = 250
learning_rate = 0.1

def feedVariablesToModel(x, weights, biases):

    # Encoder Hidden layer with sigmoid activation #1
    encoder1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    encoder2 = tf.nn.sigmoid(tf.add(tf.matmul(encoder1, weights['encoder_h2']), biases['encoder_b2']))

    # Decoder Hidden layer with sigmoid activation #1
    decoder1 = tf.nn.sigmoid(tf.add(tf.matmul(encoder2, weights['decoder_h1']), biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    decoder2 = tf.nn.sigmoid(tf.add(tf.matmul(decoder1, weights['decoder_h2']), biases['decoder_b2']))

    return decoder2

loss = tf.losses.mean_squared_error(x, decoder_op)
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

def trainModel(matrix, optimizer, loss):

    for i in range(epochs):

        avg_cost = 0
        for batch in matrix:
            _, l = session.run([optimizer, loss], feed_dict={x: batch})
            avg_cost += l
        avg_cost /= num_batches

        print("Epoch: {} Loss: {}".format(i + 1, avg_cost))

#read from properties file
p = Properties()
with open("config.properties", "rb") as f:
    p.load(f, "utf-8")

train = p.get("train").data
path = p.get("path").data

#use preprocessing module and get user-item matrix
trainData = PreProcessing.readData(train)
trainData = trainData.sort_values(['user', 'item'], ascending=[True, True])
#trainData = PreProcessing.normalizeData(trainData)
matrix, users, items = PreProcessing.buildMatrix(trainData)

# Network Parameters
num_input = len(items)
num_hidden_1 = 10               # 1st layer num features
num_hidden_2 = 5                # 2nd layer num features
tf.disable_v2_behavior()
#none arg = any no of rows, num_input arg = num_input no. of col
x = tf.placeholder(tf.float64, [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1], dtype=tf.float64)),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2], dtype=tf.float64)),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1], dtype=tf.float64)),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input], dtype=tf.float64)),
}

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1], dtype=tf.float64)),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2], dtype=tf.float64)),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1], dtype=tf.float64)),
    'decoder_b2': tf.Variable(tf.random_normal([num_input], dtype=tf.float64)),
}

# Construct model
decoder_op = feedVariablesToModel(x, weights, biases)

# Define loss and optimizer, minimize the squared error
loss = tf.losses.mean_squared_error(x, decoder_op)
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# to save the variables
saver = tf.train.Saver()

with tf.Session() as session:

    session.run(tf.global_variables_initializer())

    matrix = np.array_split(matrix, num_batches)
    trainModel(matrix, optimizer, loss)

    saved_path = saver.save(session, path)





