from __future__ import print_function
from __future__ import division
import os
import csv
import numpy as np
import tensorflow as tf


# from alexnet import AlexNet
# from datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator
from tensorflow import ConfigProto
from victorinox import victorinox
# from mynet import mynet
import time
import shutil
import pandas as pd
from victorinox import victorinox
# from alexnet import AlexNet
from sklearn.metrics import precision_recall_curve

import numpy as np
import tensorflow as tf
from datagenerator import ImageDataGenerator

class tf_knn(object):
    def __init__(self):
        return

    def get_data(self,csv_file="/a.csv",sep=",",class_idx=-1):
        df=pd.read_csv(csv_file,sep=sep,header=None)
        npd=np.array(df)
        X=npd[:,:class_idx]
        Y = npd[:, class_idx]
        return X,Y

    def knn_test(self,train_csv="/a.csv",
                 test_csv="/a.csv",
                 batch_size=32,
                 num_classes=21,
                 shuffle=True,
                 mean_pixels=[127,127,127]):
        tool=victorinox()

        # with tf.device('/gpu:0'):#with tf.device('/cpu:0'):
        #     tr_data = ImageDataGenerator(train_csv,
        #                                  mode='training',
        #                                  batch_size=batch_size,
        #                                  num_classes=num_classes,
        #                                  shuffle=shuffle,
        #                                  mean_pixels=mean_pixels)
        #
        #     val_data = ImageDataGenerator(test_csv,
        #                                   mode='inference',
        #                                   batch_size=batch_size,
        #                                   num_classes=num_classes,
        #                                   shuffle=False,
        #                                   mean_pixels=mean_pixels)
        #     # create an reinitializable iterator given the dataset structure
        #     iterator = Iterator.from_structure(tr_data.data.output_types,
        #                                        tr_data.data.output_shapes)
        #     next_batch = iterator.get_next()
        #     training_init_op = iterator.make_initializer(tr_data.data)
        #     validation_init_op = iterator.make_initializer(val_data.data)
            # Import MNIST data
        # from tensorflow.examples.tutorials.mnist import input_data
        # mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

        # In this example, we limit mnist data
        Xtr, Ytr = self.get_data(train_csv,sep=",")#mnist.train.next_batch(5000)  # 5000 for training (nn candidates)
        Xte, Yte = self.get_data(test_csv,sep=",")#mnist.test.next_batch(200)  # 200 for testing
        # tf Graph Input
        xtr = tf.placeholder(tf.float32, [None, np.shape(Xtr)[1]])
        xte = tf.placeholder(tf.float32, np.shape(Xte)[1])#[784])

        from tensorflow.contrib.bayesflow.python.ops.csiszar_divergence import jensen_shannon

        # Nearest Neighbor calculation using L1 Distance
        # Calculate L1 Distance

        #distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
        distance=tool.get_average_of_JensenShannon_using_tensorflow(xtr,xte)
        # Prediction: Get min distance index (Nearest neighbor)
        pred = tf.arg_min(distance, 0)

        accuracy = 0.

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Start training
        with tf.Session() as sess:

            # Run the initializer
            sess.run(init)

            # loop over test data
            for i in range(len(Xte)):
                # Get nearest neighbor
                nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
                # Get nearest neighbor class label and compare it to its true label
                print("Test", i, "Prediction:", np.argmax(Ytr[nn_index]), \
                      "True Class:", np.argmax(Yte[i]))
                # Calculate accuracy
                if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
                    accuracy += 1. / len(Xte)
            print("Done!")
        print("Accuracy:", accuracy)