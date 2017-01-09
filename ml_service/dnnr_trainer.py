from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function
from sklearn import cross_validation
from sklearn import metrics
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.contrib import learn

import sys
import numpy as np
import pandas as pd
from ml_common import *
from IPython import display


CSV_FILE_PATH = '''./test.csv'''
if len(sys.argv) > 1:
    CSV_FILE_PATH = sys.argv[1]

CSV_FILE_FORMAT = {
    'zipcode': str, 
    'longitude': np.float32, 
    'latitude': np.float32,
    'is_for_sale': bool,
    'property_type': str,
    'bedroom': np.float32,
    'bathroom': np.float32,
    'size': np.float32,
    'list_price': np.float32,
    'last_update': np.float32
}
ODNN_MODEL_OUTPUT_DIR = "./odnnmodel/"

# Set the output display to have one digit for decimal places, for display readability only.
pandas.options.display.float_format = '{:.1f}'.format

# Load in the data from CSV files.
property_dataframe = pandas.read_csv(CSV_FILE_PATH, dtype=CSV_FILE_FORMAT)
#print property_dataframe.head(10)
# Randomize the index.
property_dataframe = property_dataframe.reindex(
   np.random.permutation(property_dataframe.index))

# Pick out the columns we care about.
property_dataframe = property_dataframe[COLUMNS]


# Clean up data
property_dataframe = property_dataframe[property_dataframe['is_for_sale'] == True]
property_dataframe = property_dataframe[property_dataframe['list_price'] != 0]
property_dataframe = property_dataframe[property_dataframe['size'] != 0]
property_dataframe = property_dataframe[property_dataframe['bedroom'] != 0]
# Drop rows with any value NaN
property_dataframe = property_dataframe.dropna()
x = property_dataframe[UCOLUMNS]
y = property_dataframe[PRICE_COLUMN]
#print x.head(10)
#print x.head(20)
x_train, x_test, y_train, y_test = cross_validation.train_test_split(
      x, y, test_size=0.2, random_state=42)

  # Scale data (training set) to 0 mean and unit standard deviation.
  #scaler = preprocessing.StandardScaler()
  #x_train = scaler.fit_transform(x_train)


  # Build 2 layer fully connected DNN with 10, 10 units respectively.feature_columns = learn.infer_real_valued_columns_from_input(x_train)

#regressor = learn.DNNRegressor(
 #    feature_columns=feature_columns, hidden_units=[10, 10],model_dir=DNN_MODEL_OUTPUT_DIR)
estimator =learn.DNNRegressor(
    feature_columns=feature_columns_dnn,
    hidden_units=[10,10],
    optimizer=tf.train.ProximalAdagradOptimizer(
      learning_rate=0.1,
      l1_regularization_strength=0.001),
      model_dir=ODNN_MODEL_OUTPUT_DIR
    )
 # Fit
def input_fn_train():
    return input_xy_fn(x_train, y_train)

print "fit..........."
estimator.fit(input_fn=input_fn_train,steps=2000)

def input_fn_test():
    return input_xy_fn(x_test,y_test)

print estimator.evaluate(input_fn=input_fn_test, steps=100)
print "Evaluation done!"

# Predict and score
#_predicted = list(
#   estimator.predict(input_fn=input_fn_test, as_iterable=True))
#print y_predicted
#print "score...."
#score = metrics.mean_squared_error(y_predicted, y_test)

#print('MSE: {0:f}'.format(score))

# Let's make predicitions on that training data.
sample = pandas.DataFrame({'zipcode': '94015', 'property_type': 'Single Family', 'bedroom': 4, 'bathroom': 2, 'size': 1500, 'list_price':0}, index=[0])
def input_fn_predict():
  return input_fn(sample)

prediction = estimator.predict(input_fn=input_fn_predict)
print prediction