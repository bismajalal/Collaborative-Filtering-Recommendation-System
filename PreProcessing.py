import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

def readData(dataset):
    df = pd.read_csv(dataset)
    return df

def normalizeData(df):
    # Normalize in [0, 1]
    r = df['rating'].values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(r.reshape(-1, 1))
    df['rating'] = x_scaled
    return df

def buildMatrix(df):

    # Convert DataFrame in user-item matrix with ratings as entries
    matrix = df.pivot(index='user', columns='item', values='rating')
    matrix.fillna(0, inplace=True)
    users = matrix.index.tolist()
    items = matrix.columns.tolist()
    return matrix.to_numpy(), users, items

def main():
    df = pd.read_csv('ratings_small.csv', names=['user', 'item', 'rating', 'timestamp'])
    df = df.drop('timestamp', axis=1)
    df = df.drop([0], axis=0)
    trainData, testData = train_test_split(df, test_size=0.2)
    trainData.to_csv('train.csv')
    testData.to_csv('test.csv')

if __name__ == '__main__':
    main()