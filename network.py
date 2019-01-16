import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim

class Network(object):
    def __init__(self):
        self.img_size = 256

        #### feature vector size ###

        self.feature_vector_size  = ''  ## to be defined later


        ####    sizes of filters   ####
        size_filter_1 = 3
        size_filter_2 = 3
        size_filter_3= 3

        ####   end of sizes of filters   ####

        ####    number of filters  ####
        no_of_filters_1 = 64
        no_of_filters_2 = 64
        no_of_filters_3 = 64

        ####    end of number of filters   ####

        #### number of strides   ####

        self.stride_1 = 1
        self.stride_2 = 1
        self.stride_3 = 1

        ####    end of number of strides   ####

        #### fully connected layer units ###

        self.hidden_units1 = 64
        self.hidden_units2 = 32    ### final feature vector size

        self.train_path=''
        self.test_path = ''
        self.val_path = ''                  #validation path

        ####    defining filters   ####


        self.filter_1 = tf.get_variable("filter1", shape=[size_filter_1, size_filter_1, 3, no_of_filters_1],
                                        dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        self.filter_2 = tf.get_variable("filter2", shape=[size_filter_2, size_filter_2, no_of_filters_1, no_of_filters_2],
                                        dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        self.filter_3 = tf.get_variable("filter3", shape=[size_filter_3, size_filter_3, no_of_filters_2, no_of_filters_3],
                                        dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        ####    end of filters   ####

        self.x = tf.placeholder(dtype= tf.float32, name ='x', shape=[None,self.img_size,self.img_size,3])
        self.y = tf.placeholder(dtype=tf.float32, name='y', shape=[None, self.feature_vector_size])


    def create_network(self):
        ksize = [1,3,3,1]
        ## layer1 ##

        x=tf.layers.batch_normalization(self.x)
        x = tf.nn.conv2d(input=x, filter=self.filter_1, strides=[1, self.stride_1, self.stride_1, 1],
                             padding="SAME", name="conv1")
        x=tf.nn.relu(x,name="relu1")
        x=tf.nn.max_pool(x,ksize=ksize,strides=[1,2,2,1],padding="VALID",name='pool1')

        ## end of layer1 ##

        ## layer2 ##

        x = tf.layers.batch_normalization(self.x)
        x = tf.nn.conv2d(input=x, filter=self.filter_2, strides=[1, self.stride_2, self.stride_2, 1],
                             padding="SAME", name="conv2")
        x = tf.nn.relu(x, name="relu2")
        x = tf.nn.max_pool(x, ksize=ksize, strides=[1, 2, 2, 1], padding="VALID", name='pool2')

        ## end of layer2 ##

        ## layer3 ##

        x = tf.layers.batch_normalization(self.x)
        x = tf.nn.conv2d(input=x, filter=self.filter_3, strides=[1, self.stride_3, self.stride_3, 1],
                             padding="SAME", name="conv3")
        x = tf.nn.relu(x, name="relu3")
        x = tf.nn.max_pool(x, ksize=ksize, strides=[1, 2, 2, 1], padding="VALID", name='pool3')

        ## end of layer3 ##

        #### fully connected layers ####

        x = tf.contrib.layers.flatten(x)
        ## FC1 ##

        x = slim.fully_connected(x, self.hidden_units1)

        ## FC2 ##
        x = slim.fully_connected(x, self.hidden_units2) ### output

        return x

    def train_network(self,anchor_output,positive_output,negative_output):

        x_reference =
        x_train =
        margin =


        #### Triplet  loss function ###
        d_pos = tf.reduce_sum(tf.square(anchor_output - positive_output), 1)
        d_neg = tf.reduce_sum(tf.square(anchor_output - negative_output), 1)

        loss = tf.maximum(0., margin + d_pos - d_neg)
        loss = tf.reduce_mean(loss)

        ### end loss function ###

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2).minimize(self.cost)











