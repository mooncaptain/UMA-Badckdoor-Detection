__author__ = 'mooncaptain'
import math
import torch
import cv2
import numpy as np
import random
from PIL import Image


def gamma_transform(img):
    gamma = random.uniform(0.4, 1.6)
    img = (img)**(gamma)
    return (img)


def blur_transform(img):
    k = random.randrange(1,5,2)
    return cv2.bilateralFilter(img,k,20,20)

def addweight_transform(img):
    h, w, ch = img.shape
    # zero_matrix
    src2 = np.zeros([h, w, ch], img.dtype)
    a=random.uniform(0.8, 1.4)
    g=random.uniform(-0.2, 0.2)
    img_out = cv2.addWeighted(img, a, src2, 1 - a, g)  #  #g=brightness, a=contrast
    return (img_out)

def rotate_transform(img, angle):
    rows,cols,ch=img.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
 #   print(rotation_matrix.shape)
    return cv2.warpAffine(img, rotation_matrix, (cols, rows))