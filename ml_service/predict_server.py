import numpy as np
import pandas as pd
import pyjsonrpc
import tensorflow as tf
import time
from tensorflow.contrib import learn
from ml_common import *
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

SERVER_HOST = 'localhost'
SERVER_PORT = 5050

MODEL_DIR = './model'
ODNN_MODEL_OUTPUT_DIR = './odnnmodel'
MODEL_UPDATE_LAG = 5


def reload_dnn_regressor_model():
   estimator =learn.DNNRegressor(
    feature_columns=feature_columns_dnn,
    hidden_units=[10,10],
    optimizer=tf.train.ProximalAdagradOptimizer(
      learning_rate=0.1,
      l1_regularization_strength=0.001),
      model_dir=ODNN_MODEL_OUTPUT_DIR
    )
   return dnn_regressor

#linear_regressor = tf.contrib.learn.LinearRegressor(
#    feature_columns=feature_columns,
#    model_dir=MODEL_DIR)

estimator =learn.DNNRegressor(
    feature_columns=feature_columns_dnn,
    hidden_units=[10,10],
    optimizer=tf.train.ProximalAdagradOptimizer(
      learning_rate=0.1,
      l1_regularization_strength=0.001),
      model_dir=ODNN_MODEL_OUTPUT_DIR
    )

print "Model loaded"

class ReloadModelHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        # Reload model
        print "Model update detected. Loading new model."
        time.sleep(MODEL_UPDATE_LAG)
        reload_dnn_regressor_model
        print "Model updated."

class RequestHandler(pyjsonrpc.HttpRequestHandler):
    """Test method"""
    @pyjsonrpc.rpcmethod
    def predict(self, zipcode_dnn, property_type_dnn, bedroom, bathroom, size):
        sample_dnn = pandas.DataFrame({
            'zipcode': zipcode_dnn,
            'property_type': property_type_dnn,
            'bedroom': bedroom,
            'bathroom': bathroom,
            'size': size, 
            'list_price':0},index=[0])
           
        print "prediting dnn....."
        def input_fn_predict():
            return input_fn(sample_dnn)
        prediction = estimator.predict(input_fn=input_fn_predict)
        print prediction
        return prediction[0].item()

# Setup watchdog
observer = Observer()
observer.schedule(ReloadModelHandler(), path=MODEL_DIR, recursive=False)
observer.start()

# Threading HTTP-Server
http_server = pyjsonrpc.ThreadingHttpServer(
    server_address = (SERVER_HOST, SERVER_PORT),
    RequestHandlerClass = RequestHandler
)

print "Starting predicting server ..."
print "URL: http://" + str(SERVER_HOST) + ":" + str(SERVER_PORT)

http_server.serve_forever()
