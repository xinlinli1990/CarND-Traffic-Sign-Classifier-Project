import numpy as np
import cv2

def data_augmentation(X_train, y_train, target):
    """
    This function receive training dataset X_train and labels y_train. 
    Args:
		X_train: Training dataset
		y_train: Training dataset labels
		target: Target number of training data for each label
    Returns:
        ret_X_train: New training dataset after data augmentation
		ret_y_train: New training dataset labels after data augmentation
    Raises:
    """
    
    labels, counts = np.unique(y_train, return_counts=True)
    
    X_aug = []
    y_aug = []
    
    for label, count in zip(labels, counts):
        augmentation_num = target - count
        X_train_label = X_train[y_train == label]

        for i in range(augmentation_num):
            # pick a random image from X_train_label
            img = X_train_label[np.random.randint(count)]
            # image augmentation
            intensity = 0.75
            img = single_image_augmentation(img, intensity)
            
            X_aug.append(img)
            y_aug.append(label)
    
    # Append new data to X_train.    
    ret_X_train = np.concatenate((X_train, np.array(X_aug)), axis=0)
    ret_y_train = np.concatenate((y_train, np.array(y_aug)), axis=0)

    return ret_X_train, ret_y_train

def data_augmentation_without_original_data(X_train, y_train, target):
    
    labels, counts = np.unique(y_train, return_counts=True)
    
    X_aug = []
    y_aug = []
    
    for label, count in zip(labels, counts):
        augmentation_num = target
        X_train_label = X_train[y_train == label]

        for i in range(augmentation_num):
            # pick a random image from X_train_label
            img = X_train_label[np.random.randint(count)]
            # image augmentation
            intensity = 0.75
            img = single_image_augmentation(img, intensity)
            # append to 
            X_aug.append(img)
            y_aug.append(label)
            
    ret_X_train = np.array(X_aug)
    ret_y_train = np.array(y_aug)

    return ret_X_train, ret_y_train

def single_image_augmentation(img, intensity):
    """
    This function receive one original image and randomly generate one new image based on given intensity
    Args:
        img: input image
        intensity: intensity of rotation and projective transform
    Returns:
        img: output image
    Raises:
    """

    img = rotate_img(img, intensity)
    img = projective_transform(img, intensity)

    return img

from skimage.transform import rotate
from skimage.transform import warp
from skimage.transform import ProjectiveTransform
from numpy import random

def rotate_img(img, intensity):
    delta = 30. * intensity 
    img = rotate(img, random.uniform(-delta, delta), mode='edge')
    return img

def projective_transform(img, intensity):
    #print(img.dtype)
    #print(np.max(img))
    #print(np.min(img))
    image_x_size = img.shape[0]
    image_y_size = img.shape[1]

    dx = image_x_size * 0.3 * intensity
    dy = image_y_size * 0.3 * intensity

    tl_top = random.uniform(-dy, dy)
    tl_left = random.uniform(-dx, dx)
    bl_bottom = random.uniform(-dy, dy)
    bl_left = random.uniform(-dx, dx)
    tr_top = random.uniform(-dy, dy)
    tr_right = random.uniform(-dx, dx)
    br_bottom = random.uniform(-dy, dy)
    br_right = random.uniform(-dx, dx)

    transform = ProjectiveTransform()
            
    transform.estimate(
        np.array((
            (tl_left, tl_top),
            (bl_left, image_x_size - bl_bottom),
            (image_y_size - br_right, image_x_size - br_bottom),
            (image_y_size - tr_right, tr_top))),
        np.array((
            (0, 0),
            (0, image_x_size),
            (image_y_size, image_x_size),
            (image_y_size, 0))))    

    img = warp(img, transform, mode='edge')
    img = (img * 255).astype(np.uint8)
    #print(img.dtype)
    #print(np.max(img))
    #print(np.min(img))
    return img