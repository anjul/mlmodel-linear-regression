import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Argument parser for command line readability
parser = argparse.ArgumentParser()
parser.add_argument("features", type=str, help="comma separated column names to use as features columns (col1,col2,..)")
parser.add_argument("label",    type=str, help="a column name to use as label column")
parser.add_argument("csv_file", type=str, help="an absolute file path of the csv file")
parser.add_argument("-v", "--verbosity", action="store_true", help="display parsed input arguments")

# Process command line arguments
args = parser.parse_args()
features = args.features.split(',')
label = args.label
csv_file = args.csv_file

if args.verbosity:
    print("Inputs")
    print("--------------------------------------")
    print("Features : {}".format(features))
    print("Label : {}".format(label))
    print("CSV File : {}".format(csv_file))
    print("\n")

# Read csv file into pandas dataframe
df = pd.read_csv(csv_file)

# Split the dataframe into features and label
X = df[features]
Y = df[label]

# Split the dataset to train and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

# Fit the model with train dataset
lin_reg_model = LinearRegression().fit(X_train, Y_train)

# model evaluation for training set
y_train_predict = lin_reg_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print()

# model evaluation for testing set
y_test_predict = lin_reg_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))