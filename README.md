# Collaborative-Filtering-Recommendation-System
A Movie Recommendation System which takes user-movie data and returns personalized movies recommendations for each user.

Collaborative Filtering recommendation systems give recommendations to a user based on the preferences of users similar to him/her. Similar users are the ones which give similar ratings to same movies.

Technologies/Packages used:
•	Pycharm
•	Tensorflow
•	Sklearn
•	Pandas
•	Numpy
•	jproperties

Solution: A user-movie matrix whose entries are the ratings given by the users to the movies is fed into the model. For each user, the model returns the predicted rating for each movie. The movies with the highest predicted ratings are chosen as the recommendation for each user.
This is a deep unlearning, unsupervised problem. Autoencoder with two hidden layers is used as implemented here. save() and restore() functions from Tensorflow’s Saver class are used to save and restore variables.

Data: Dataset used for this model can be found here. The file used is ‘ratings_small.csv’. It is split with a ratio 80:20 into train and test dataframes which are written into ‘train.csv’ and ‘test.csv’ files. 

Execution Guidelines
1. Make relevant changes in the properties file i.e update the path of dataset.csv, train.csv and test.csv. For 'path', put in the path to folder where you want you model to be saved.

2. Run the Driver.py file. It will ask you to input user IDs and will give recommendations for those users.

3. If dataset.csv is not split already or if you want to split it again, uncomment line 10 in Driver.py i.e PreProcessing.main()

4. If you have already trained the model and do not want to do it again, comment line 13 in the Driver.py i.e import Training



