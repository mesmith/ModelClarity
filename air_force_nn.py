#!/usr/bin/python

#mod ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ### Model Clarity
#
# This is a simple neural network model in pandas and pytorch.
# It takes an arbitrary dataset consisting of categorical and continuous variables, and allows us
# to designate one column as the label, and a set of other columns as features that
# may or may not predict the labels.
#
# The takeaway from this model is to answer a fairly common objection that python models in general, 
# and pytorch models specifically, fundamentally lack transparency.
#
# This has negative effects in the sense that data science developer labor--not training efficiency, nor production efficiency--is typically the most expensive component in any machine learning project.
#
# I believe that this labor cost can be vastly improved by:
#
# * better documentation within models, whereby the data scientist stops assuming that there are no consumers of the program text; and
#
# * a bias towards assuming that the next reader of the program text is not expert in all of the details of pandas, numpy, tensors, and pytorch.
#
# The goal of this model is to demonstrate that it is possible to write a pytorch model that a reader with a modest understanding of machine learning can understand without having to refer to online searches, courseware, or face to face conversation.
#
# Ideally, you can use this model as a machine learning workbench.
# I'll call it "knowledge transfer learning", where hopefully you can convert your
# knowledge of machine learning into learning how pytorch works.
#
# Notably, this model splits the data into the "holy trinity" of 
# training, cross validation, and testing data, as well as calculating and
# displaying losses, accuracy, precision, recall, and F1* scores.
#


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# For command line arguments
#
import sys
import argparse

# For saving the category encoders for use in production.
#
# The normalizations that are performed on categorical
# variables during training must work exactly the same 
# as when we perform inference on the variables in production.
#
import category_encoders as ce
import pickle

# For saving the model
#
import io
import torch.onnx
import numpy as np
import math
import pandas as pd
from pymongo import MongoClient

# For sharing variables and functions with production (inference).
# You should go ahead and have a look at this module, as it contains feature
# and neural network model definitions.
#
import air_force_shared as afs
import os

# Command line arguments.  We allow a single argument that points to
# a configuration file.  The config file is itself written in python.
# By default, the configuration file is "default_config.py".
#
# Example usage:
#  $ python air-force-nn.py                        # uses default_config.py
#  $ python air-force-nn.py --config mongo_config  # uses mongo_config.py
#  $ python air-force-nn.py --config csv_config    # uses csv_config.py
#
# The configuration file lets us specify our data source, as well as
# some model parameters.
#
# If you run this via jupyter notebook, it will only ever use the default_config.py;
# if you want to change the model behaviour in juypter, then change default_config.py first.
#
default_args = {
  "config": "default_config"
}
parser = argparse.ArgumentParser(description="Training arguments")
parser.add_argument('-c', '--config', help='Name of configuration file, defaults to default_config.py')
#
# This argument is here only so that jupyter notebook will work.
#
parser.add_argument('-f', '--file', help='Temporary file name for jupyter notebook.')
arglist = parser.parse_args()
if (arglist.config == None):
    arglist.config = default_args['config']
if os.path.isfile(arglist.config + '.py') != True:
    print('Sorry, there is no configuration file named ' + arglist.config + '.py.')
    sys.exit(2)

# Read the configuration based on the --config option
#
import importlib
cfg = importlib.import_module(arglist.config)
config = cfg.config

# Save the configuration in a well-known pickle file, so at inference time,
# it will be easy to find.
#
# This will save the object 'obj' to disk at file 'target'
#
def save_to_disk(obj, target):
    filehandler = open(target, 'wb')
    pickle.dump(obj, filehandler)
    filehandler.close()
save_to_disk(config, afs.get_file_location(afs.CONFIG_FILE))

# +
# A utility for making a connection to mongo.
#
# Returns null if 'db' does not exist in the mongodb instance.
# Will fail if host/port don't specify a mongodb endpoint, or
# if username/password are needed and are incorrect.
#
def _connect_mongo(host, port, username, password, db):
    if username and password:
        mongo_uri = 'mongodb://%s:%s@%s:%s/%s' % (username, password, host, port, db)
        conn = MongoClient(mongo_uri)
    else:
        conn = MongoClient(host, port)

    return conn[db] if (db in conn.list_database_names()) else None

# This will read a MongoDB collection in batches, avoiding some out-of-memory
# errors for very large collections
#
def batched(cursor, batch_size):
    batch = []
    for doc in cursor:
        batch.append(doc)
        if batch and not len(batch) % batch_size:
            yield batch
            batch = []

    if batch:   # last documents
        yield batch

# Read from MongoDB and store into pandas dataframe.
#
# Return null if the database or collection does not exist.
#
def read_mongo(db, collection, query={}, host='localhost', port=27017, username=None, password=None, no_id=True):
    df = pd.DataFrame()

    # Connect to MongoDB
    #
    db = _connect_mongo(host=host, port=port, username=username, password=password, db=db)
    if db == None:
        return df

    # Make sure the collection exists
    #
    if collection not in db.list_collection_names():
        return df

    # Make a query to the specific DB and Collection
    #
    cursor = db[collection].find(query)

    # Expand the cursor and construct the DataFrame
    # Use batching for large datasets
    #
    for batch in batched(cursor, 10000):
        df = df.append(batch, ignore_index=True)

    # Delete the _id
    #
    if no_id:
        del df['_id']

    return df

# -

# Read CSV or MongoDB dataset into raw_data.
#
if config['data']['type'] == 'CSV':
    raw_data = pd.read_csv(config['data']['collection'])
else:                                               # Only MongoDB, for now
    raw_data = read_mongo(config['data']['dbname'], config['data']['collection'],
        config['data']['query'], config['data']['host'], config['data']['port'],
        config['data']['username'], config['data']['password'], config['data']['noid'])
if raw_data.empty:
    print('No data found.  Recheck your collection and database connect string.')
raw_data


# Initialize accumulated training and test results, allowing us
# to manually aggregate results for display and analysis.
#
all_results = {
    'all_training': [],
    'all_validation': [],
    'all_test': []
}

# +
# Convert raw 'data' into just the data that contains the features.
#
# We need 'label_field' to be part of the returned data, since it's
# possible that rows will be dropped due to nulls,
# and we need to keep the feature
# columns correlated with the label column.
#
def get_feature_data(data, features, label_field):
    all_fields = np.concatenate((features, [label_field]))

    return data[all_fields].copy()

def get_data_without_nulls(data, features, label_field):
    
    # Get just the slice of the data frame containing all of the features
    # and the label
    #
    feature_data = get_feature_data(data, features, label_field)

    # Drop rows with missing values for any of the features.
    #
    # We don't want to retain any useless pandas index information, and
    # the old indexes actually cause problems when doing pd.concat(),
    # so get rid of them.
    #
    # Before doing this, we'll convert empty strings to 'np.nan'.  Otherwise,
    # dropna() will treat empty strings as non-null data.
    #
    # NOTE: We're making a global assumption here that an empty string implies a 
    # missing value.  This may not be true for every dataset.
    #
    empty_string_replaced = feature_data.replace('', np.nan, inplace=False)
    data_without_nulls = empty_string_replaced.dropna()
    data_without_nulls.reset_index(drop=True, inplace=True)

    return data_without_nulls

# Return an array that encodes the mean and standard deviation for all
# of the 'config['collection']['cont_features']' within 'data'.  The result will be persisted and
# reused during production.
#
def get_continuous_encoder(cont_features, data):

    # Get the mean and standard deviation of all values of 'field'
    #
    res = []
    for column in cont_features:
        s = pd.Series(data[column])
        numerics = pd.to_numeric(s, errors='coerce')
        mean = numerics.mean()
        std = numerics.std()
        res += [{'column': column, 'mean': mean, 'std': std}]

    return res

# +
# Create a one hot encoder and save it to disk.  That will allow it to work
# exactly the same in production.
#
# Note that we must only encode the feature data, so the shape of the data is the
# same for training and production.  So we remove the labels prior to encoding:
# in production, there is no label!
#
features = (config['collection']['cat_features'] + config['collection']['cont_features'])
data_without_nulls = get_data_without_nulls(raw_data, features, config['collection']['label_field'])
feature_data_without_nulls = data_without_nulls[features]
cat_enc = ce.OneHotEncoder(cols=config['collection']['cat_features'], use_cat_names=True, return_df=True).fit(feature_data_without_nulls)
with_one_hot = cat_enc.transform(feature_data_without_nulls)

save_to_disk(cat_enc, afs.get_file_location(afs.CATEGORICAL_ENCODER_FILE))

label_data = data_without_nulls[config['collection']['label_field']]
with_one_hot_and_label = pd.concat([with_one_hot, label_data], axis=1)

# Add in the continuous features.  First we calculate the mean and standard deviation
# for each continuous feature, and then save them to the afs.CONTINUOUS_ENCODER_FILE,
# so we can use the exact same normalizer during production.  Then, do the normalization.
#
cont_enc = get_continuous_encoder(config['collection']['cont_features'], with_one_hot_and_label)
save_to_disk(cont_enc, afs.get_file_location(afs.CONTINUOUS_ENCODER_FILE))

encoded_data = afs.fit_normalized(with_one_hot_and_label, cont_enc, afs.norm_prefix)
encoded_data


# +
# Given dataset, return vector of labels with 1 indicating success
# and 0 indicating failure
#
def get_labels(data, field, true_values):
    true_map = {k: True for k in true_values}

    return tuple([1 if i in true_map else 0 for i in data[field]])

# Convert the designated categorical label to a boolean, indicating
# successful or failed recruitment.  We put the new column in the
# first position so it's easy to see.  And do it immutably.
#
def add_labels(data, labels):
    label_column = pd.DataFrame({afs.generic_label: labels})

    return pd.concat([label_column, data], axis=1)

labels = get_labels(encoded_data, config['collection']['label_field'], config['collection']['true_values'])
with_labels = add_labels(encoded_data, labels).drop([config['collection']['label_field']], axis=1)

print('# of successes=', (with_labels[afs.generic_label] == 1).sum())
print('# of failures=', (with_labels[afs.generic_label] == 0).sum())
with_labels



# +
# This will return a new data set with its rows randomized.
# The indices in the result will be reset to be ordered from 0.
#
def randomize(data_set):
    random_data_set = data_set.copy().sample(frac=1).reset_index(drop=True)

    return random_data_set

# Now we have a dataset suitable for a machine learning model.
#
# Split the data into the training data,
# the cross-validation set, and the test set.
#
# Prior to splitting, shuffle the data so we do not accidentally
# learn a random, ungeneralizable pattern on one data sequence.
#
def split_data(data):
    # This randomly shuffles the data, immutably.
    #
    shuffled = randomize(data)

    # Split the data into 60% training, 20% validation, 20% test.
    #
    n_sample = shuffled.shape[0]
    first = n_sample * 6 // 10
    second = n_sample * 8 // 10

    training_data = shuffled[:first]
    training_data.reset_index(drop=True, inplace=True)

    validation_data = shuffled[first:second]
    validation_data.reset_index(drop=True, inplace=True)

    test_data = shuffled[second:]
    test_data.reset_index(drop=True, inplace=True)

    return {
        'training_data': training_data,
        'validation_data': validation_data,
        'test_data': test_data
    }

splits = split_data(with_labels)

training_data = splits['training_data']
validation_data = splits['validation_data']
test_data = splits['test_data']

print('training data length =', training_data.shape[0])
print('validation data length =', validation_data.shape[0])
print('test data length =', test_data.shape[0])
# -

print(' % success in training set: {:.4f}'.format(100*training_data[afs.generic_label].sum() / len(training_data)), '%')
print(' % success in validation set: {:.4f}'.format(100*validation_data[afs.generic_label].sum() / len(validation_data)), '%')
print(' % success in test set: {:.4f}'.format(100*test_data[afs.generic_label].sum() / len(test_data)), '%')

# +
# Now that we know the number of features, we can define
# the hyperparameters of the network
#
# When creating hyperparameters, don't count the output label column in
# the training data
#
n_columns = training_data.shape[1] - 1

# There are several practices regarding the setting of the number of
# hidden layers.  Heaton Research suggests that the number of hidden
# layers should be 2/3 the size of input, plus size of output.
# We'll use that here.
#
n_classes = 2
n_hidden = (n_columns * 2 // 3) + n_classes

hyper = {
    'input_size': n_columns,
    'hidden_size': n_hidden,
    'num_classes': n_classes,
    'dropout_prob': .2,
}

# Save hyperparameters to disk
#
save_to_disk(hyper, afs.get_file_location(afs.HYPER_FILE))
hyper


# -

# In the more general cases, a pytorch model uses a data loader,
# which yields batches of (potentially streaming) data.
#
# We'll create one for this dataset in order to demonstrate the
# capability.
#
# In this data loader, we assume that data_set is a pandas dataframe,
# and the data_set includes labels in label_field.  We will return
# a batch of features and a batch of labels in the result.  And we'll
# cast the result batches as tensors.
#
# The yielded results are tensors: column information from the dataframe
# is removed.
#
def data_loader(data_set, label_field, batch_size=1, shuffle=False):

    rds = randomize(data_set) if shuffle else data_set

    # First, split the pandas data_set into a labels vector and a features vector
    #
    feature_data_frame = rds.loc[:, data_set.columns != label_field]
    feature_data = feature_data_frame.values.tolist() # to python list

    label_data_frame = rds[label_field]
    label_data = label_data_frame.values.tolist() # to python list

    # We'll return only complete batches, throwing away the remainder
    # data past the last even multiple of batch_size.
    #
    n_batches = len(feature_data) // batch_size

    for i in range(n_batches):
        idx = i * batch_size
        x = feature_data[idx : idx + batch_size]
        y = label_data[idx : idx + batch_size]
    
        # Labels have to be long values in order for the NLLLoss()
        # function to work properly.
        #
        yield torch.FloatTensor(x), torch.LongTensor(y)

# +
# Define our neural network.  Note that we will use LogSoftmax
# as the output calculator, as we are generating probabilities
# of selecting values of a class.  In turn, LogSoftmax is best
# paired with the NLLLoss(), or negative log likelihood loss, function
# as the loss function, which needs LogSoftmax as the output
# calculator.
#
# Note also the use of a dropout layer, which will regularize the
# data during training.  The dropout layer is not applied during
# evaluation (that is, after calling model.eval()).
#
# We comment this out, because we are actually generating this in
# the air_force_shared module for later use in production.
#
# model = torch.nn.Sequential(
    # torch.nn.Linear(hyper['input_size'], hyper['hidden_size']),
    # torch.nn.ReLU(),
    # torch.nn.Dropout(p = hyper['dropout_prob']),
    # torch.nn.Linear(hyper['hidden_size'], hyper['num_classes']),
    # torch.nn.LogSoftmax(dim=1))
model = afs.get_model()

model
# -

# Use the negative log loss function, paired with LogSoftmax() above.
#
loss_fn = torch.nn.NLLLoss()
loss_fn

# Optimizer.  We'll just use stochastic gradient descent.
#
optimizer = optim.SGD(model.parameters(), lr=config['model']['learning_rate'])
optimizer

# Check for CUDA GPU support
#
use_cuda = torch.cuda.is_available()
print("CUDA check done:  use_cuda=", use_cuda)
device = torch.device('cuda' if use_cuda else 'cpu')
device

# We will do some learning curve plotting during the training and
# validation modeling below.  This will allow us to see if the
# model works at all (training loss improves), if it overfits
# (validation loss gets worse), or if there's a "knee" in the validation
# curve where we should stop running the model.
#
# %matplotlib inline
import matplotlib.pyplot as plt


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
def get_predicted_labels(output):
    predicted_probs = torch.exp(output)            
    _, top_class = predicted_probs.topk(1, dim=1)

    return torch.flatten(top_class)

# Given labels and predictions, both tensors,
# return true/false positive/negatives.
#
def get_stats(labels, predictions):

    # True Positive (TP): we predict a label of 1 (positive),
    # and the true label is 1.
    # TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    #
    TP = torch.sum(((predictions == 1) & (labels == 1)).type(torch.LongTensor))
 
    # True Negative (TN): we predict a label of 0 (negative), 
    # and the true label is 0.
    #
    # TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
    TN = torch.sum(((predictions == 0) & (labels == 0)).type(torch.LongTensor))
 
    # False Positive (FP): we predict a label of 1 (positive),
    # but the true label is 0.
    #
    # FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    FP = torch.sum(((predictions == 1) & (labels == 0)).type(torch.LongTensor))
 
    # False Negative (FN): we predict a label of 0 (negative),
    # but the true label is 1.
    #
    # FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
    FN = torch.sum(((predictions == 0) & (labels == 1)).type(torch.LongTensor))

    return TP.item(), TN.item(), FP.item(), FN.item()

def get_precision(tp, fp):
    if tp + fp == 0:
        return 1
    return tp / (tp + fp)

def get_recall(tp, fn):
    if tp + fn == 0:
        return 1
    return tp / (tp + fn)

# Given true/false positive/negatives, return f1 score
#
def get_f1(tp, tn, fp, fn):
    precision = get_precision(tp, fp)
    recall = get_recall(tp, fn)
    if precision + recall == 0:
        return 1
    return 2 * (precision * recall) / (precision + recall)


# -

# Perform a single training epoch.
# Return the training loss for that epoch.
#
def train(model, loader, data_len, label, optimizer, loss_fn):
    training_loss = 0.0    # accumulated training loss during this epoch

    # This will change internal model state so that gradients are
    # automatically calculated
    #
    model.train()

    for features, labels in loader:
        # Reset the optimizer's gradients...otherwise, they will
        # (incorrectly) accumulate from one batch to the next one
        #
        optimizer.zero_grad()

        # Transfer to GPU
        #
        features, labels = features.to(device), labels.to(device)

        # Get model results 
        #
        output = model(features)

        # Calculate the loss, and back-propagate it
        #
        loss = loss_fn(output, labels)
        loss.backward()

        # Recalculate all of the weights in the model based on the
        # loss function's gradient
        #
        optimizer.step()

        training_loss += loss.data.item()

    # Since we will be comparing aggregated losses between training
    # and validation, we will need to get the averages, as the
    # number of records likely differ between training and validation.
    #
    training_loss /= data_len
    print('AVG TRAINING LOSS={:.4f}'.format(training_loss))

    return training_loss

# Perform validation test
#
def validate(model, loader, data_len, label, loss_fn, test_type):
    valid_loss = 0.0
    num_correct = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    # Set model to eval(), which disables dropout.
    # Turn off gradients during validation.
    #
    # Also, we use enumerate() to get the batch index; this is helpful if
    # we want to display a few intermediate results without making the model
    # extremely slow.
    #
    model.eval()
    with torch.set_grad_enabled(False):
        for i, (features, labels) in enumerate(loader):
            features, labels = features.to(device), labels.to(device)
    
            output = model(features)
    
            loss = loss_fn(output, labels)
            valid_loss += loss.data.item()
        
            # Convert the output into a tensor that has the same
            # shape as the labels.  Then calculate the number of
            # correct labels.
            #
            predicted_labels = get_predicted_labels(output)
            equals = (predicted_labels == labels).type(torch.LongTensor)
            num_correct += torch.sum(equals)
        
            tp, tn, fp, fn = get_stats(labels, predicted_labels)
            true_positives += tp
            true_negatives += tn
            false_positives += fp
            false_negatives += fn
    
            # Debug: look at output vs. labels to verify shapes, 
            # statistics, etc.
            #
            # if i <= 10:
                # print('for TEST TYPE=', test_type)
                # print('  labels=', labels)
                # print('  predicted labels=', predicted_labels)
                # print('  tp=', tp, ', tn=', tn, ', fp=', fp, ', fn=', fn)
                # print('  true_positives=', true_positives, ', true_negatives=', true_negatives)
                # print('  false_positives=', false_positives, ', false_negatives=', false_negatives)
                # print('  validation output=', output)
                # print('  validation labels=', labels)
                # print('  equals=', equals)
            
        valid_loss /= data_len
        accuracy = num_correct.item() / data_len
        precision = get_precision(true_positives, false_positives)
        recall = get_recall(true_positives, false_negatives)
        f1 = get_f1(true_positives, true_negatives, false_positives, false_negatives)
        # print('     ', test_type, ' loss={:.4f}'.format(valid_loss))
        # print('     ', test_type, ' accuracy=', num_correct.item()/data_len)
        # print('     ', test_type, ' true positives=', true_positives)
        # print('     ', test_type, ' true negatives=', true_negatives)
        # print('     ', test_type, ' false positives=', false_positives)
        # print('     ', test_type, ' false negatives=', false_negatives)
        # print('     ', test_type, ' precision=', precision)
        # print('     ', test_type, ' recall=', recall)
        # print('     ', test_type, ' f1=', f1)

        return {
            'valid_loss': valid_loss, 
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

# Each epoch will read the entire training and validation dataset.
# During training, the loss function will be used to recalculate the
# weights within the neural network.
#
# Then, after each epoch, calculate the effect of the model on the
# validation set.
#
def train_and_test(model, training_data, validation_data, test_data,\
                   label, num_epochs, batch_size, optimizer, loss_fn):

    # Use the GPU if it's available
    #
    model = model.to(device)

    # Track losses across epochs so we can see if learning is occurring
    #
    all_training = []
    all_validation = []
    all_test = []

    for epoch in range(num_epochs):
        print('*****\nepoch =', epoch)

        train_loader = data_loader(training_data, label, batch_size)
        training_loss = train(model, train_loader, len(training_data), label,
                              optimizer, loss_fn)
        all_training.append(training_loss)

        # Perform cross-validation test
        #
        validation_loader = data_loader(validation_data, label, batch_size = batch_size)
        validation_stats = validate(model, validation_loader, len(validation_data),\
                                    label, loss_fn, 'validation')
        all_validation.append(validation_stats)
        # print('    validation_stats=', validation_stats)

        # Perform generalization test
        #
        test_loader = data_loader(test_data, label, batch_size = batch_size)
        test_stats = validate(model, test_loader, len(test_data), label,\
                              loss_fn, 'test')
        all_test.append(test_stats)
        # print('    test_stats=', test_stats)

    return {
        'all_training': all_training,
        'all_validation': all_validation,
        'all_test': all_test
    }

# This function will return the result of merging prior results with
# new ones.  This thus allows us to manually re-run the model
# (using jupyter) rather than re-running from scratch with a higher
# num_epochs setting.
#
def append_results(old_results, new_results):
    all_training = np.concatenate((old_results['all_training'], new_results['all_training']))
    all_validation = np.concatenate((old_results['all_validation'], new_results['all_validation']))
    all_test = np.concatenate((old_results['all_test'], new_results['all_test']))

    return {
        'all_training': all_training,
        'all_validation': all_validation,
        'all_test': all_test,
    }

# Run the model.
#
# Note that if you run this particular cell multiple times,
# it will just continue where it left off, as the model retains its state.
# Typically, if you do that, you'll see evidence of overfitting in the
# graph below!
#
# If you want to rerun the model from scratch, just start running from the
# cell where the model is originally created, above.
#
results = train_and_test(model, training_data, validation_data, test_data,\
                        afs.generic_label, config['model']['num_epochs'], config['model']['batch_size'],
                        optimizer, loss_fn)
all_results = append_results(all_results, results)

# Display the model's dictionary, which contains the learned model parameter values.
# This will be saved for later serving.
#
model.state_dict()

# +
# Save the model.  Note that the model is a mutable object: at this point, it
# should be trained.
#
# First, save in native pytorch (Python pickle) format.
#
torch.save(model.state_dict(), afs.get_file_location(afs.MODEL_FILE))

# Next, save in interoperable ONNX format.  The dummy_input is just
# an example input of the right shape.  We use some generic parameter
# transforms to generate the right shape, then put the dummy input
# on the GPU if possible.
#
# As of this writing, ONNX should still be considered experimental, as it
# does not support all possible pytorch operations (for example, operations
# using the ATen module).
#
# I'm including this ONNX model saver just for experimentation.  Comment it
# out if it causes problems.
#
shape_of_first_layer = list(model.parameters())[0].shape
shape_of_first_layer
N,C = shape_of_first_layer[:2]
dummy_input = torch.Tensor(N, C)
dummy_input = dummy_input.cuda() if use_cuda else dummy_input
torch.onnx.export(model, dummy_input, afs.get_file_location(afs.ONNX_MODEL_FILE))

# +
# This returns an array containing only the values for 'key'
#
def get_one_key(array, key):
    return [x[key] for x in array]

# Show plot indicating training and validation loss.
# As long as the validation loss is going down, the model is learning.
# If the training loss is going down, but validation loss isn't, the model
# is overfitting.
#
plt.plot(all_results['all_training'], label='Training loss')
plt.plot(get_one_key(all_results['all_validation'], 'valid_loss'), label='Validation loss')
plt.plot(get_one_key(all_results['all_test'], 'valid_loss'), label='Test loss')
plt.legend()

plt.subplot(1, 2, 1)
plt.title('Validation')
plt.plot(get_one_key(all_results['all_validation'], 'accuracy'), label='Accuracy')
plt.plot(get_one_key(all_results['all_validation'], 'precision'), label='Precision')
plt.plot(get_one_key(all_results['all_validation'], 'recall'), label='Recall')
plt.plot(get_one_key(all_results['all_validation'], 'f1'), label='F1')
plt.legend()
plt.subplot(1, 2, 2)
plt.title('Test')
plt.plot(get_one_key(all_results['all_test'], 'accuracy'), label='Accuracy')
plt.plot(get_one_key(all_results['all_test'], 'precision'), label='Precision')
plt.plot(get_one_key(all_results['all_test'], 'recall'), label='Recall')
plt.plot(get_one_key(all_results['all_test'], 'f1'), label='F1')
plt.legend()
plt.tight_layout()
plt.show()

# ### Result Analysis
#
# For the example data, we see that there does appear to be at least some minor predictability of recruitment success from the sex, race, and state features, as evidenced by the fact that the validation and test losses do shrink a bit as the model attempts to learn the relationship.
#
# However, the accuracy is mediocre in extremis, and the F1 score is not great either.  You would most likely not want to change a recruitment policy based on the results.
#
# Another point:  Often this model will start by just predicting negatives, and will gradually start
# picking positives.  This manifests itself by showing 100% precision, 0% recall, and 0% F1.
# For the above model, just re-running the model (without restarting it) will often cause it to
# start predicting positives.
