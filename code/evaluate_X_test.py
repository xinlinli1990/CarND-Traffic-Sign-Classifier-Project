import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from load_data import load_data
from data_augmentation import data_augmentation, data_augmentation_without_original_data
from data_preprocessing import data_preprocessing
import models 

model = models.convNet
BATCH_SIZE = 128
# Path to trained network
restore_path = './debug/m=convNet2 lr=0.0005 do=0.75 l2=0.0001/Best_Solution'
    
X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()

pre_X_test, pre_y_test = data_preprocessing(X_test, y_test)

tf.reset_default_graph()

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
l2_reg_const = tf.placeholder(tf.float32)
phase = tf.placeholder(tf.bool, name='phase')
one_hot_y = tf.one_hot(y, 43)


logits = model(x, keep_prob, l2_reg_const, phase)


cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
# add regularization
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss_operation = loss_operation + tf.add_n(reg_losses)

# For batch normalization
update_operation = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)

with tf.control_dependencies(update_operation):
    training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def evaluate(X_data, y_data, restore_path, l2_reg_const_param=0.0, isTraining=False):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
    
        sess.run(tf.global_variables_initializer())
        
        saver.restore(sess, restore_path)
    
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
            
            accuracy, loss = sess.run([accuracy_operation, loss_operation], 
                                      feed_dict={x: batch_x, 
                                                 y: batch_y, 
                                                 keep_prob: 1.0, 
                                                 l2_reg_const: l2_reg_const_param, 
                                                 phase: isTraining})
            total_loss += (loss * len(batch_x))
            total_accuracy += (accuracy * len(batch_x))
            
    return (total_accuracy / num_examples), (total_loss / num_examples)
    
    
accu, loss = evaluate(pre_X_test, pre_y_test, restore_path)

print("accuracy = " + str(accu))
print("loss = " + str(loss))