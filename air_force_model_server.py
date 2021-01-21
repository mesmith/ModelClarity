# Run with:
#  $  FLASK_ENV=development FLASK_APP=air_force_model_server.py flask run
#
# Note.  This will run the server twice, as a feature of the reloader.
#
from flask import Flask, jsonify, request
import numpy as np
import math
import pandas as pd
import io
import torch
import torch.nn as nn
import torch.nn.functional as F

# For reusing the same categorical encoder from training
#
import category_encoders as ce
import pickle

# Variables and functions shared with training
#
import air_force_shared as afs

# For reading the model via URL
#
import os
import sys
from urllib.request import urlopen
from shutil import copyfileobj
from tempfile import NamedTemporaryFile

# For reading command line
#
import argparse

# +
# This version will load the model from a file passed in from the environment,
# or from the current directory if the environment variable is missing.
#
# Loading from the environment will scale better during production,
# as we can point the server to any model by just changing the environment.
#
def load_model_from_file():
    m = afs.get_model()
    model_params_location = afs.get_file_location(afs.MODEL_FILE)

    # Note the use of the CPU here.  Currently, the docker image does not
    # support CUDA. FIXME: It should be possible to do a GPU at runtime.
    #
    m.load_state_dict(torch.load(model_params_location, map_location='cpu'))
    return m

# This model loader will fetch the model from a URL in the environment.
# Used if this server is to be hosted on a cloud service.
#
def load_model_from_url():
    m = afs.get_model()
    model_params_url = afs.get_url_location(afs.MODEL_FILE)
    with urlopen(url) as fsrc, NameTemporaryFile() as fdst:
        copyfileobj(fsrc, fdst)
        # Note the use of the CPU here.  Currently, the docker image does not
        # support CUDA. FIXME: It should be possible to do a GPU at runtime.
        #
        m.load_state_dict(torch.load(fdst, map_location='cpu'))
    return m

# Load the model.  If there is a missing environment variable,
# or the model couldn't be loaded for any reason,
# just stop the server entirely.
#
model = load_model_from_file()
print('Loaded model:', model)
if model == None:
    sys.exit()

# Set the model's state to 'eval()' so it won't run
# backprop or use dropout layers (they are only used during training).
#
model.eval()

# Transform input data into a tensor, to be used for inference.
#
# Note: We are assuming input is a table, not a single row.
# 
# Similar to the function on the trainer, but does not include labels.
#
def transform_input(data, cat_features, cont_features, norm_prefix):

    # This will fail if any of the continuous variables are missing.
    #
    features = (cat_features + cont_features)
    feature_data = data[features].copy()

    # Deal with nulls on input.
    #
    empty_string_replaced = feature_data.replace('', np.nan, inplace=False)
    data_without_nulls = empty_string_replaced.dropna()
    data_without_nulls.reset_index(drop=True, inplace=True)

    # Convert the categorical feature values into one-hot vectors, removing
    # the original categorical feature columns.  Reuse the same categorical
    # encoder that we trained on.
    #
    cat_enc = afs.read_object(afs.get_file_location(afs.CATEGORICAL_ENCODER_FILE))
    with_one_hot = cat_enc.transform(data_without_nulls.copy())

    # The continuous encoder was written by us, so its methods aren't
    # exactly consistent with the categorical data encoder.
    #
    cont_enc = afs.read_object(afs.get_file_location(afs.CONTINUOUS_ENCODER_FILE))

    # Convert the continuous feature values into normalized features,
    # removing the original continuous feature columns.
    #
    with_normalized = afs.fit_normalized(with_one_hot, cont_enc, norm_prefix)

    encoded_list = with_normalized.values.tolist()  # to python list

    return torch.FloatTensor(encoded_list)  # to tensor

# +
# The output is a vector of log_softmax (that is, a vector
# of log-probabilities over the class).  We'll convert 
# these to just a single prediction for each label.
#
# (This is exactly the kind of calculation that is often
# glossed over in models, where the modeler assumes that
# the reader somehow knows by looking at an obscure function
# what it's supposed to represent.)
#
# The shape of both of the results of topk() is [batch_size, 1],
# so we flatten down to just [batch_size], so the result
# can be compared against the labels.
#
def get_probabilities(model_prediction):
    return torch.exp(model_prediction)            

# Return the probabilities for each label prediction
#
def get_each_label_probability(model_prediction):
    predicted_probs = get_probabilities(model_prediction)            
    _, top_class = predicted_probs.topk(1, dim=1)

    return torch.flatten(top_class)

# Return the index within a vector of labels that indicates the prediction
#
def get_prediction_index(model_prediction):
    predicted_labels = get_each_label_probability(model_prediction)
    return predicted_labels.item()

# Return the human-readable (textual) prediction
#
def get_human_prediction(model_prediction, collection):
    prediction_index = get_prediction_index(model_prediction)
    return collection['label_outputs'][prediction_index]

def get_json_response(input_tensor, collection):
    model_prediction = model(input_tensor)
    probs = get_probabilities(model_prediction).detach().numpy()[0].tolist()
    return jsonify({
        'predict_index': get_prediction_index(model_prediction),
        'predict_human': get_human_prediction(model_prediction, collection),
        'probabilities': probs
    })

# This is the entrypoint when deploying.
# See run-model-service.sh.
#
def create_app():
    app = Flask(__name__)

    # Get the configuration for datastore and model.
    #
    config = afs.read_object(afs.get_file_location(afs.CONFIG_FILE))
    print("Inference service was started.")

    @app.route("/")
    def status():
        return jsonify({"status": "ok"})

    @app.route('/gethyper', methods=['GET', 'POST'])
    def gethyper():
        hyper = afs.read_object(afs.get_file_location(afs.HYPER_FILE))
        return jsonify({'hyper': hyper})

    @app.route('/getmodel', methods=['GET', 'POST'])
    def getmodel():
        # This is the function that print() uses.
        #
        modelrep = model.__repr__()
        return jsonify({'model': modelrep})

    # The prediction stub just generates a zero tensor to
    # test the model
    #
    @app.route('/predict_zeroes', methods=['GET', 'POST'])
    def predict_zeroes():
        hyper = afs.read_object(afs.get_file_location(afs.HYPER_FILE))
        input_tensor = torch.zeros(1, hyper.input_size)  # one row of columns with zeroes
        return get_json_response(input_tensor, config['collection'])

    @app.route('/predict_ones', methods=['GET', 'POST'])
    def predict_ones():
        hyper = afs.read_object(afs.get_file_location(afs.HYPER_FILE))
        input_tensor = torch.ones(1, hyper.input_size)  # one row of columns with ones
        return get_json_response(input_tensor, config['collection'])

    @app.route('/predict_rand', methods=['GET', 'POST'])
    def predict_rand():
        hyper = afs.read_object(afs.get_file_location(afs.HYPER_FILE))
        input_tensor = torch.rand(1, hyper.input_size)  # one row of columns with random values
        return get_json_response(input_tensor, config['collection'])

    # Allow the POST input to be either an array of samples, or a single row.
    # Currently, only works for one row.  FIXME.
    #
    @app.route('/predict', methods=['GET', 'POST'])
    def predict():
        content = request.get_json()
        if content != None:
            list_content = content if type(content) is list else [content]

            # The Docker image may throw a deprecation warning, like this:
            #   ... pandas.io.json.json_normalize is deprecated,
            #       use pandas.json_normalize instead
            # However, don't make that change until the non-docker environment
            # uses the same pandas version as the docker image.
            #
            raw_data = pd.io.json.json_normalize(list_content)
            input_tensor = transform_input(raw_data,
                config['collection']['cat_features'],
                config['collection']['cont_features'], afs.norm_prefix)
            return get_json_response(input_tensor, config['collection'])
        else:
            return jsonify({
                'predict_index': None,
                'predict_human': None,
                'probabilities': None
            })

    return app
