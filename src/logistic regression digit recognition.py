import pandas as pd

from sklearn.linear_model import LogisticRegression

# Import the training and test data files:
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#Separate the data values from the labels
train_val = train.loc[:, 'pixel0':]
test_val = test.loc[:, 'pixel0':]

#Normalize (inspired by Kaggle user tommyttf's suggestion)
train_val = train_val / 255.0
test_val = test_val / 255.0

#Run the regression to train the algorithm
alg = LogisticRegression(penalty = 'l2', solver = 'lbfgs', multi_class = 'multinomial') # can also try simple logreg instead
alg.fit(train_val, train.loc[:, 'label'])

#Output the predictions and consolidate the results into a csv file
predictions = alg.predict(test)

results = pd.DataFrame({"ImageId": range(1, test.shape[0] + 1),
                        "label": predictions})

results.to_csv("predictions.csv", index=False)
