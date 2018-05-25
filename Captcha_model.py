# -*- coding: utf-8 -*
import tensorflow as tf
import math

class captchaModel():
    def __init__(self,
                 width = 160,
                 height = 60,
                 char_num = 4,
                 classes = 62):
        self.width = width
        self.height = height
        self.char_num = char_num
        self.classes = classes

    def conv2d(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev = 0.1)
        weights = tf.Variable(initial)
        return weights

    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def create_model(self,x_images,keep_prob):
        #first layer
        with tf.name_scope('first_layer'):
            weight_conv1 = self.weight_variable([5,5,1,32])
            bias_conv1 = self.bias_variable([32])
            pool_conv1 = self.max_pool_2x2(tf.nn.relu(tf.nn.bias_add(self.conv2d(x_images,weight_conv1), bias_conv1)))
            dropout_conv1 = tf.nn.dropout(pool_conv1,keep_prob)

        #second layer
        with tf.name_scope('second_layer'):
            weight_conv2 = self.weight_variable([5,5,32,64])
            bias_conv2 = self.bias_variable([64])
            pool_conv2 = self.max_pool_2x2(tf.nn.relu(tf.nn.bias_add(self.conv2d(dropout_conv1,weight_conv2), bias_conv2)))
            dropout_conv2 = tf.nn.dropout(pool_conv2,keep_prob)


        #third layer
        with tf.name_scope('third_layer'):
            weight_conv3 = self.weight_variable([5,5,64,64])
            bias_conv3 = self.bias_variable([64])
            pool_conv3 = self.max_pool_2x2(tf.nn.relu(tf.nn.bias_add(self.conv2d(dropout_conv2,weight_conv3), bias_conv3)))
            dropout_conv3 = tf.nn.dropout(pool_conv3,keep_prob)


        #first fully layer
        with tf.name_scope('full_connect_layer'):
            input_FC1 = tf.reshape(dropout_conv3, [-1,20*8*64])
            weight_FC1 = self.weight_variable([20*8*64,1024])
            bias_FC1 = self.bias_variable([1024])
            pool_FC1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(input_FC1,weight_FC1), bias_FC1))
            dropout_FC1 = tf.nn.dropout(pool_FC1,keep_prob)

        #second fully layer
        with tf.name_scope('final_layer'):
            weight_Final = self.weight_variable([1024,62*4])
            bias_Final = self.bias_variable([62*4])
            y_conv = tf.add(tf.matmul(dropout_FC1,weight_Final),bias_Final)

        return y_conv
