import cv2
import numpy as np
from sklearn.utils import shuffle

def data_normalization(X_train, y_train):
    """
    Args:
        X_train: input images
        y_train: input labels
    Returns:
        ret_X_train: output images
        ret_y_train: output labels
    Raises:
    """
    
    ret_X_train = (X_train.astype(float) - 128) / 256
    ret_y_train = y_train
    
    return ret_X_train, ret_y_train

def data_preprocessing(X_train, y_train, is_shuffle=True):
    """
    This function convert all input images into grayscale, then apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance traffic sign feature.
    Args:
        X_train: input images
        y_train: input labels
    Returns:
        X_train: output images
        y_train: output labels
    Raises:
    """

    tmp_X_train = []
    
    for i in range(X_train.shape[0]):
        img = X_train[i]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        tmp_X_train.append(img[:,:,np.newaxis])
    
    X_train = np.array(tmp_X_train)

    X_train, y_train = data_normalization(X_train, y_train)

    if is_shuffle:
        X_train, y_train = shuffle(X_train, y_train)

    return X_train, y_train


