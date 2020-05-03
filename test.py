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

