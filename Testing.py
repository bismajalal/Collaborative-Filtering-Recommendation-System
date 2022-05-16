import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from sklearn.metrics import mean_squared_error
from math import sqrt
import copy
from jproperties import Properties
import PreProcessing

k = 10
num_batches = 250

def preProcessing(file):

    #read from properties file
    p = Properties()
    with open("config.properties", "rb") as f:
        p.load(f, "utf-8")
    train = p.get("train").data     #we need dimensions of placeholder from train data
    trainData = PreProcessing.readData(train)
    users = trainData.user.nunique()
    items = trainData.item.nunique()

    path = p.get("path").data
    test = p.get(file).data

    #use preprocessing module and get user-item matrix
    testData = PreProcessing.readData(test)
    testData = testData.sort_values(['user', 'item'], ascending=[True, True])

    return path, testData, users, items

def restoreVariables(session, path):

    saver = tf.train.import_meta_graph(path + '.meta')
    saver.restore(session, path)

    weights = {
        'encoder_h1': tf.get_default_graph().get_tensor_by_name('Variable:0'),
        'encoder_h2': tf.get_default_graph().get_tensor_by_name('Variable_1:0'),
        'decoder_h1': tf.get_default_graph().get_tensor_by_name('Variable_2:0'),
        'decoder_h2': tf.get_default_graph().get_tensor_by_name('Variable_3:0'),
    }

    biases = {
        'encoder_b1': tf.get_default_graph().get_tensor_by_name('Variable_4:0'),
        'encoder_b2': tf.get_default_graph().get_tensor_by_name('Variable_5:0'),
        'decoder_b1': tf.get_default_graph().get_tensor_by_name('Variable_6:0'),
        'decoder_b2': tf.get_default_graph().get_tensor_by_name('Variable_7:0'),
    }

    return weights, biases

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

def addColumnsToMatrix(nullCols, testMatrix, testData, num_batches):

    #change dimensions of testMatrix to feed into the model to get predictions
    temp = np.zeros((testData.user.nunique(), nullCols))
    newMatrix = np.hstack((testMatrix, temp))
    #print(newMatrix.shape)
    #newMatrix = np.array_split(newMatrix, num_batches)
    return newMatrix

def getPredictionMatrix(session, x, matrix, decoder_op, dummyCols):

    predictions = pd.DataFrame()
    #matrix = np.concatenate(matrix, axis=0)

    preds = session.run(decoder_op, feed_dict={x: matrix})

    #delete dummy cols from prediction matrix
    realCols = preds.shape[1] - dummyCols
    predsMatrix = np.delete(preds, slice(realCols, preds.shape[1]+1), 1)

    # change from matrix to dataframe with movieIDs as cols
    predictions = predictions.append(pd.DataFrame(predsMatrix))
    # make 3 colums: user/item/rating
    predictions = predictions.stack().reset_index(name='rating')
    predictions.columns = ['user', 'item', 'rating']
    #print(predictions.shape)
    return predictions

def getRMSE(predictions, data, tempTest):

    # print the predictions that are a part of test data
    keys = ['user', 'item']
    i1 = predictions.set_index(keys).index
    i2 = data.set_index(keys).index
    recs = predictions[i1.isin(i2)]
    # recs = recs.sort_values(['user', 'rating'], ascending=[True, False])
    # recs = recs.groupby('user').head(k)
    #print(recs.head(20))

    ratings = recs['rating'].tolist()
    actual = tempTest['rating'].tolist()

    # print(actual)
    # print(ratings)

    print(sqrt(mean_squared_error(actual, ratings)))

def main(file):

    path, testData, users, items = preProcessing(file)
    tempTest = copy.deepcopy(testData)
    #testData = PreProcessing.normalizeData(testData)
    testMatrix, testUsers, testItems = PreProcessing.buildMatrix(testData)

    # make users and testusers equal in length so we can map testUsers to the users in tested predictions
    for i in range(users - len(testUsers)):
        testUsers.append(None)
    # make items and testItems equal in length so we can map testItems to the items in tested predictions
    for i in range(items - len(testItems)):
        testItems.append(None)


    tf.disable_v2_behavior()
    x = tf.placeholder(tf.float64, [None, items])

    with tf.Session() as session:

        weights, biases = restoreVariables(session, path)
        decoder2 = feedVariablesToModel(x, weights, biases)

        #add dummy columns to matrix to make no. of cols = no. of cols in placeholder
        dummyCols = items - testMatrix.shape[1]
        newMatrix = addColumnsToMatrix(dummyCols, testMatrix, testData, num_batches)
        predictions = getPredictionMatrix(session, x, newMatrix, decoder2, dummyCols)
        # map user id to user index in the dataframe
        predictions['user'] = predictions['user'].map(lambda value: testUsers[value])
        # map movie id to movie index in the dataframe
        predictions['item'] = predictions['item'].map(lambda value: testItems[value])
        predictions['rating'] = predictions['rating'] * 5
        #print(predictions)

        #df = df.drop('timestamp', axis=1)
        #remove the movies already watched
        keys = ['user', 'item']
        i1 = predictions.set_index(keys).index
        i2 = testData.set_index(keys).index
        recs = predictions[~i1.isin(i2)]

        #store the movies already watched for evaluation
        checkpreds = predictions[~i1.isin(i2)]

        recs = recs.sort_values(['user', 'rating'], ascending=[True, False])
        recs = recs.groupby('user').head(k)
        #print(testData.shape)
        #print(predictions.shape)
        #print(recs.head(20))
        #print(recs)
        recs.to_csv('predictions.csv', mode='a', header=False)

        print("RMSE for ", file, "file:")
        getRMSE(predictions, testData, tempTest)

if __name__ == '__main__':
    main(file)

