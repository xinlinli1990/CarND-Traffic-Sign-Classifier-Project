import numpy as np
import cv2
from skimage.transform import rotate
from skimage.transform import warp
from skimage.transform import ProjectiveTransform
from numpy import random

def single_image_augmentation(img, intensity, label):

    img = rotate_img(img, intensity)
    img = projective_transform(img, intensity)

    return img, label

def rotate_img(img, intensity):
    delta = 30. * intensity 
    img = rotate(img, random.uniform(-delta, delta), mode='edge')
    return img

def projective_transform(img, intensity):
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
    return img

from joblib import Parallel, delayed
    
def parallel_test(X_train, y_train, target):
    
    labels, counts = np.unique(y_train, return_counts=True)
    
    X_aug = []
    y_aug = []
    
    for label, count in zip(labels, counts):
        augmentation_num = target - count
        X_train_label = X_train[y_train == label]

        parallel_res = Parallel(n_jobs=4)(delayed(single_image_augmentation)(X_train_label[np.random.randint(count)], 0.75, label=label) for i in range(augmentation_num))

        print(parallel_res)
        print(len(parallel_res))