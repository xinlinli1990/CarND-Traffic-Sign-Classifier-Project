from load_data import load_data
from parallel_data_augmentation import parallel_test
from data_preprocessing import data_preprocessing
import matplotlib.pyplot as plt
import models
import cv2

from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf

data_augmentation_target = 3

X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()

# unique, counts = np.unique(y_train, return_counts=True)
# plt.figure(1)
# plt.bar(unique, counts)
# plt.xlabel('traffic sign label')
# plt.ylabel('images count')
# plt.show()

X_train, y_train = shuffle(X_train, y_train)
X_train_test = X_train[0:20]
y_train_test = y_train[0:20]

for i in range(len(X_train_test)):
    plt.imsave('./test_image/ori'+str(i)+'.jpg', X_train_test[i])

augmented_X_train, augmented_y_train = parallel_test(X_train_test, y_train_test, data_augmentation_target)

for i in range(len(augmented_X_train)):
    plt.imsave('./test_image/aug'+str(i)+'.jpg', augmented_X_train[i])
# print(np.max(X_train))
# print(np.min(X_train))
# print(np.max(augmented_X_train))
# print(np.min(augmented_X_train))


# pre_X_train, pre_y_train = data_preprocessing(augmented_X_train, augmented_y_train)
# print(pre_X_train.shape)
# for i in range(len(pre_X_train)):
    # plt.imsave('./test_image/pre'+str(i)+'.jpg', pre_X_train[i, :, :, 0])
    
    
# print(np.max(X_train))
# print(np.min(X_train))
# print(np.max(augmented_X_train))
# print(np.min(augmented_X_train))
# print(np.max(pre_X_train))
# print(np.min(pre_X_train))

# unique, counts = np.unique(pre_y_train, return_counts=True)
# plt.figure(1)
# plt.bar(unique, counts)
# plt.xlabel('traffic sign label')
# plt.ylabel('images count')
# plt.show()

#pre_X_valid, pre_y_valid = data_preprocessing(X_valid, y_valid)

# plt.imshow(X_train[1357])
# plt.show()

#image = np.concatenate((X_train[1357], pre_X_train[1357]), axis=0)
# plt.imshow(pre_X_train[1357])
# plt.show()

