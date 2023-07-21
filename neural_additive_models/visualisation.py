import matplotlib as plt
import data_utils
import graph_builder
import numpy as np
from absl import flags
from tensorflow import compat as tf
import tensorflow as tensorf
import nam_train
import os
import tempfile
import sys
import pandas as pd
import h5py
FLAGS = flags.FLAGS

FLAGS(sys.argv)

tf.v1.disable_v2_behavior()

house_x, house_y, column_names = data_utils.load_dataset('Housing')

"""""
#APPROACH 1: I modified the training function in nam_train.py to return the nam_model. 
#The idea is to then acess the models featureNNs in a session that has the same graph 
#as in the training function in nam_train.py

def create_and_train_nam(dataset_name, regression):
    FLAGS.training_epochs = 8
    FLAGS.save_checkpoint_every_n_epochs = 2
    FLAGS.early_stopping_epochs = 4
    FLAGS.dataset_name = dataset_name
    FLAGS.regression = regression
    FLAGS.num_basis_functions = 16
    logdir = './NamCode/google-research/nam_models'
    tf.v1.gfile.MakeDirs(logdir)
    data_gen, _ = nam_train.create_test_train_fold(fold_num=1)
    (x_train, y_train), (x_validation, y_validation) = next(data_gen)

    return nam_train.training(x_train, y_train, x_validation, y_validation, logdir)

nam_model = create_and_train_nam('Housing', True)[2]

with tf.v1.get_default_graph() as graph:
    with tf.v1.Session(graph = graph) as session:
        saver = tf.v1.train.import_meta_graph('./NamCode/google-research/nam_models/model/model.ckpt-112.meta')
        saver.restore(session, tf.v1.train.latest_checkpoint('./NamCode/google-research/nam_models/model'))
        session.run(nam_model.calc_outputs(house_x[0])) #calc_outputs returns the output of each featureNN and here it receives the first row of data

#We get the following error -> "Graph is invalid, contains a cycle with 8 nodes, including: 
#feature_nn/StatefulPartitionedCall, feature_nn_1/StatefulPartitionedCall, feature_nn_2/StatefulPartitionedCall"

"""""

"""""

#APPROACH 2: Create a new model and then either load the weights onto it with Model.load_weights or run the new model in a session.
#saver.restore should make it so that session.run computes the model with the weights and biases of a trained model. 

train_data, val_data = data_utils.split_training_dataset(house_x, house_y, 2, stratified=False)
x_batch, (train_init_op, test_init_op) = graph_builder.create_iterators((train_data[0][0], val_data[0][0]), 1024)

nam_model = graph_builder.create_nam_model(train_data[0][0], 0.5)

with tf.v1.get_default_graph() as graph:
    with tf.v1.Session(graph = graph) as session:
        #operations_and_tensors, metrics = graph_builder.build_graph(train_data[0][0], train_data[0][1], val_data[0][0], val_data[0][0],)
        nam_model.build(train_data[0][0].shape)
        saver = tf.v1.train.import_meta_graph('./NamCode/google-research/nam_models/model/model.ckpt-112.meta')
        saver.restore(session, tf.v1.train.latest_checkpoint('./NamCode/google-research/nam_models/model_0'))
        #nam_model.load_weights(tf.v1.train.latest_checkpoint('./NamCode/google-research/nam_models/model_0')) #
        session.run(nam_model.calc_outputs(house_x[0]))

#line 69 error -> NotImplementedError: Streaming restore not supported from name-based checkpoints
#when graph building. File a feature request if this limitation bothers you.
#As a workaround, consider either using tf.train.Checkpoint to load name-based checkpoints or enabling eager execution.

#line 70 error -> Could not find variable model/activation_layer_7/c. Running graph.get_collection('variables') we see that the graph has a variable
#model_0/activation_layer_7/c. I am not sure if the issue would be fixed if I somehow name my new model model_0 or if I rename the variable names that are saved in training

"""