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

FLAGS = flags.FLAGS

FLAGS(sys.argv)

#Load the data. data_and target is a dictionary with the following key value pairs ('X', data), ('y', target) and ('problem', the type of ML problem.)
#data_and_target = data_utils.load_california_housing_data()



def create_and_train_nam(dataset_name, regression):
    FLAGS.training_epochs = 4
    FLAGS.save_checkpoint_every_n_epochs = 2
    FLAGS.early_stopping_epochs = 2
    FLAGS.dataset_name = dataset_name
    FLAGS.regression = regression
    FLAGS.num_basis_functions = 16
    logdir = os.path.join(tempfile.mkdtemp(), dataset_name)
    tf.v1.gfile.MakeDirs(logdir)
    data_gen, _ = nam_train.create_test_train_fold(fold_num=1)
    (x_train, y_train), (x_validation, y_validation) = next(data_gen)

    return nam_train.training(x_train, y_train, x_validation, y_validation, logdir)

create_and_train_nam('Housing', True)
print(FLAGS.featureNNs_outputs_values)






