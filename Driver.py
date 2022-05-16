import PreProcessing
import pandas as pd
import csv

with open('predictions.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['index','user','item', 'rating'])

# uncomment and run if your dataset is not already split into test and train
# PreProcessing.main()

# importing training runs the entire Training code. comment this line if model is already trained
import Training

# importing Testing runs nothing because all the code is in functions
import Testing

# gets predictions and write them into predictions.csv. calculates RMSE for those predictions
Testing.main('train')
Testing.main('test')

#read predictions and choose the best ones
predictions = pd.read_csv('predictions.csv')
preds = predictions.sort_values(['user', 'rating'], ascending=[True, False])
preds = preds.groupby('user').head(10)
#print(preds.head(50))

print("Predictions: ", len(predictions.item.unique()))
print("Preds: ", len(preds.item.unique()))

stop = 1
while int(stop) != 0:

    print("Total no. of users are: ", len(preds.user.unique()))
    user = input("Which user do you want recommendations for: ")
    output = preds.loc[preds['user'] == int(user)]
    recs = output['item']
    print(recs.to_string(index=False))
    stop = input("Press 0 to Exit. Press any other key to continue: ")



