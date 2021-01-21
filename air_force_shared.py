# Shared functions and variables used in both training and in production.
#
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

import os

# This is a prefix used to identify continuous variables that have been normalized.
# You probably don't need to change this.
#
norm_prefix = 'Normalized'

# This is the column name used for labels after converting the input label_field
# into 0s and 1s.  You probably don't need to change this.
#
generic_label = 'Label'

# This is where the config file is stored.
# At training time, the configuration is obtained from the command line.
# Then, the training program writes the configuration into this file,
# so at inference time, there will be no need for a command line option
# or an environment variable to point to it.
#
CONFIG_FILE = 'config.pt'

# This is where the model will be saved on disk.
# MODEL_FILE saves in standard pytorch (Python pickle) format,
# where ONNX_MODEL_FILE saves in interoperable ONNX format.
#
MODEL_FILE = 'air_force_model.pt'
ONNX_MODEL_FILE = 'air_force_model.onnx'

# This is where the categorical and continous data encoders will be saved on disk.
# We will need to run the exact same encoders when we run the model
# in production.
#
CATEGORICAL_ENCODER_FILE='af_cat_encoder.pt'
CONTINUOUS_ENCODER_FILE='af_continuous_encoder.pt'

# Save hyperparameters on disk as well, since we use the dataset's variables to
# generate the hyperparameters.
#
HYPER_FILE = 'af_hyper.pt'

# This returns the location where pickle files are.  These are created
# at training time and referenced at runtime.
#
# In addition, we see if there is an environment variable that points
# to a different location for the files, allowing this model to be retargeted.
#
def get_model_files_location():
    if "AF_MODEL_FILES" in os.environ:
        return os.environ["AF_MODEL_FILES"]
    else:
        return '.'

# Use this if we are hosting the app on a cloud service.  Currently unused.
#
def get_url_location():
    if "AF_MODEL_URL_PATH" in os.environ:
        return os.environ["AF_MODEL_URL_PATH"]
    else:
        return '.'

def get_file_location(file):
    return get_model_files_location() + "/" + file

# This returns the python object that was created at training time
#
def read_object(file_name):
    file = open(file_name, 'rb')
    obj = pickle.load(file)
    file.close
    return obj

# Return the model.  Obtain the hyperparameters from disk, then
# use them to generate the neural net.
#
def get_model():
    hyper = read_object(get_file_location(HYPER_FILE))
    return torch.nn.Sequential(
        torch.nn.Linear(hyper['input_size'], hyper['hidden_size']),
        torch.nn.ReLU(),
        torch.nn.Dropout(p = hyper['dropout_prob']),
        torch.nn.Linear(hyper['hidden_size'], hyper['num_classes']),
        torch.nn.LogSoftmax(dim=1))

# Normalize the values of the single continuous 'field'.
# A new column named "norm_prefix"_"field" will hold the normalized values.
#
def normalize(data, field, norm_prefix, mean, std):
    # Normalize using the formula: x' = (x - mean)/std.
    # Write the x' values into a field whose name has 'norm_prefix'
    #
    def norm(x, mean, std):
        return (x - mean) / std

    s = pd.Series(data[field])
    numerics = pd.to_numeric(s, errors='coerce')
    mapped = numerics.map(lambda x: 0 if math.isnan(x) else norm(x, mean, std))
    norm_field = norm_prefix + '_' + field
    out = pd.DataFrame({norm_field: mapped})

    return out

# Apply the continuous encoding in 'enc' to 'data'.
# New columns will be created for the normalized values, and the
# original continuous columns will be removed.
#
def fit_normalized(data, enc, norm_prefix):
    with_normalized = data
    for col_enc in enc:
        feature = col_enc['column']
        normalized = normalize(with_normalized, feature, norm_prefix, col_enc['mean'], col_enc['std'])
        with_normalized = pd.concat([with_normalized, normalized], axis=1).drop([feature], axis=1)

    return with_normalized
