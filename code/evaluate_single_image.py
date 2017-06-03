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


          
model = models.convNet2
learning_rate=1e-5
BATCH_SIZE=128

# image paths and their labels
list = ['./new_sample/test1.png',
        './new_sample/test2.png',
        './new_sample/test3.png',
        './new_sample/test4.png',
        './new_sample/test5.png',
        './new_sample/test6.png',
        './new_sample/test7.png']
        
labels = [7,
         23,
         11,
         22,
         30,
         28,
         23]
         
X_test = []
y_test = []

count = 0
        
for img_path in list:
    img = cv2.imread(img_path)
    X_test.append(img)
    y_test.append(labels[count])
    count += 1
    
X_test = np.array(X_test)

# Pre-processing for images
pre_X_test, pre_y_test = data_preprocessing(X_test, y_test, is_shuffle=False)

tf.reset_default_graph()

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
l2_reg_const = tf.placeholder(tf.float32)
phase = tf.placeholder(tf.bool, name='phase')
one_hot_y = tf.one_hot(y, 43)


logits = model(x, keep_prob, l2_reg_const, phase)


cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
prediction_prob = tf.nn.softmax(logits=logits)

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


def evaluate(X_data, restore_path, l2_reg_const_param=0.0, isTraining=False):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
    
        sess.run(tf.global_variables_initializer())
        
        saver.restore(sess, restore_path)
    
        probs = sess.run([prediction_prob], 
                        feed_dict={x: X_data, 
                                   keep_prob: 1.0, 
                                   l2_reg_const: l2_reg_const_param, 
                                   phase: isTraining})

    return probs
    
    
probs = evaluate(pre_X_test, './debug/m=convNet2 lr=0.0005 do=0.75 l2=0.0001/Best_Solution')

probs = probs[0]

from python_dict import sign_name_dict, sign_name_dict_short

for i in range(len(list)):
    print("image " + sign_name_dict[y_test[i]])
    prob = probs[i]
    top5_idx = prob.argsort()[-5:][::-1]
    
    top5_prob = []
    signs = []
    
    for idx in top5_idx:
        top5_prob.append(prob[idx])
        signs.append(sign_name_dict_short[idx])
        
    top5_prob.reverse()
    signs.reverse()
    
    y_pos = np.arange(len(signs))
    
    plt.close("all")
    #plt.axes([0.2,0.1,0.9,0.9])
    fig, ax = plt.subplots(1,1)
    ax.barh(y_pos, top5_prob, align='center', alpha=0.5)
    plt.yticks(y_pos, signs, size=12)
    plt.xlabel('Probability')
    plt.title('Top 5 predictions for web image ' + str(i + 1) + '\n(' + sign_name_dict[y_test[i]] + ')')
    plt.xlim([0.0, 1.0])
    fig.subplots_adjust(left = 0.2)
    #plt.tight_layout()
    #ax = plt.gca()
    #ax.yaxis.set_tick_params(labelsize=10)
    
    plt.savefig('./predict_'+str(i+1)+'.png')
    #plt.show()
    
    print(top5_idx)
    print(top5_prob)