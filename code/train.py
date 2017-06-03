from load_data import load_data
from data_augmentation import data_augmentation, data_augmentation_without_original_data
from data_preprocessing import data_preprocessing
import matplotlib.pyplot as plt
import models

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import os
import sys


def train(#model=models.LeNetWithDropout2,
          #model=models.LeNetWithDropout,
          #model=models.InceptionV3,
          model=models.convNet,
          #model=models.LeNet,
          restore_path=None,
          initial_EPOCH=0,
          learning_rate=1e-5, 
          data_augmentation_target=3000, 
          dropout_keep_prob=1.0, 
          EPOCHS=10, 
          BATCH_SIZE=128, 
          l2_reg_const_param=0.0):
    
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()
    
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
    saver = tf.train.Saver()
    best_saver = tf.train.Saver()

    def evaluate(X_data, y_data, l2_reg_const_param=0.0, isTraining=False):
        num_examples = len(X_data)
        total_accuracy = 0
        total_loss = 0
        sess = tf.get_default_session()
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
        
    debug_path = "./debug/" +\
                 "m="+str(model.__name__) +\
                 " lr="+str(learning_rate) +\
                 " do="+str(dropout_keep_prob) +\
                 " l2="+str(l2_reg_const_param)+"/"

    if not os.path.exists(os.path.dirname(debug_path)):
        os.makedirs(os.path.dirname(debug_path))
    elif restore_path is None:
        # if no restore_path
        return -1, -1
    else:
        pass

    with tf.Session() as sess:
    
    
        train_writer = tf.summary.FileWriter(debug_path+'train',
                                             sess.graph)
        test_writer = tf.summary.FileWriter(debug_path+'test')
        
        sess.run(tf.global_variables_initializer())
        
        log_file_name = debug_path+"output.log"
        if restore_path is not None:
            saver.restore(sess, restore_path)
            log_file_name = debug_path+"output_continue.log"

        with open(log_file_name, "w") as log_file:
            log_file.write("Training Hyperparameters:\n")
            log_file.write("model="+str(model.__name__)+"\n")
            log_file.write("learning_rate="+str(learning_rate)+"\n")
            log_file.write("data_augmentation_target="+str(data_augmentation_target)+"\n")
            log_file.write("dropout_keep_prob="+str(dropout_keep_prob)+"\n")
            log_file.write("l2_reg_const_param="+str(l2_reg_const_param)+"\n")
            log_file.write("EPOCHS="+str(EPOCHS)+"\n")
            log_file.write("BATCH_SIZE="+str(BATCH_SIZE)+"\n")
            log_file.write("\n")
            #print("EPOCH", "Valid Loss", "Train Loss", "Valid Accu", "Train Accu", sep='\t')
            log_file.write("EPOCH\tValid Loss\tTrain Loss\tValid Accu\tTrain Accu"+"\n")
        
        # Output loss
        loss_values = []
        max_valid_accu = -1
        diff_valid_train = 1
        
        # Use given validation dataset
        pre_X_valid, pre_y_valid = data_preprocessing(X_valid, y_valid)

        for i in range(initial_EPOCH, EPOCHS):
            if i >= 10 and max_valid_accu <= 0.10 and restore_path is None:
                return max_valid_accu, diff_valid_train
        
            # data augmentation for each EPOCH
            augmented_X_train, augmented_y_train = data_augmentation(X_train, 
                                                                     y_train, 
                                                                     data_augmentation_target)
            
            pre_X_train, pre_y_train = data_preprocessing(augmented_X_train, augmented_y_train)
            num_examples = len(pre_X_train)

            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = pre_X_train[offset:end], pre_y_train[offset:end]
                _, loss_value = sess.run([training_operation, loss_operation], 
                                         feed_dict={x: batch_x, 
                                                    y: batch_y,  
                                                    keep_prob: dropout_keep_prob, 
                                                    l2_reg_const: l2_reg_const_param, 
                                                    phase: True})

                loss_values.append(loss_value)
                
            fig = plt.figure()    
            plt.plot(loss_values)
            if restore_path is None:
                plt.savefig(debug_path+'Training_loss.png')
            else:
                plt.savefig(debug_path+'Training_loss-continue.png')
            plt.close(fig)
                    
            validation_accuracy, valid_loss = evaluate(pre_X_valid, 
                                                       pre_y_valid, 
                                                       l2_reg_const_param=l2_reg_const_param, 
                                                       isTraining=False)
            
            training_accuracy, train_loss = evaluate(pre_X_train, 
                                                      pre_y_train, 
                                                      l2_reg_const_param=l2_reg_const_param, 
                                                      isTraining=True)
            with open(log_file_name, "a") as log_file:                                       
                log_file.write(str(i+1)+"/"+str(EPOCHS)+"\t"
                               +str(valid_loss)+"\t"
                               +str(train_loss)+"\t"
                               +str(validation_accuracy)+"\t"
                               +str(training_accuracy)+"\n")
            
            if validation_accuracy > max_valid_accu:
                max_valid_accu = validation_accuracy
                diff_valid_train = training_accuracy - validation_accuracy
                best_saver.save(sess, debug_path+'Best_Solution')
                
            saver.save(sess, debug_path+'ConvNet', global_step=i)

        return max_valid_accu, diff_valid_train

