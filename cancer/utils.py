import os
import cv2
from os.path import join
from collections import defaultdict
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

from matplotlib import pyplot as plt


def read_bmp(img_path):
    return cv2.imread(img_path)

def read_dat_file(path):
    return np.loadtxt(path, delimiter=',')


def mask():
    img = cv2.imread('lena.jpg')
    mask = cv2.imread('mask.png',0)
    res = cv2.bitwise_and(img,img,mask = mask)

def convert_array_to_poly(arr):
    return arr.flatten().tolist()

def generate_mask(img, poly):
    mask = Image.new('L', img.size, 0)
    ImageDraw.Draw(mask).polygon(convert_array_to_poly(poly), outline=1, fill='#ffffff')
    return np.array(mask)

def plot_segment(img, segments):
    plt.figure(figsize=(15,15))
    plt.imshow(img)
    for poly in segments:
        plt.fill(poly[:, 0], poly[:, 1], alpha=.3, facecolor='g', edgecolor='black', linewidth=5)

    plt.axis('off')
    plt.show()

def get_min_and_max_images(img_list):
    g_min, g_max = np.inf, 0
    for img in img_list:
        cur_max, cur_min = max(img.shape[0], img.shape[1]), min(img.shape[0], img.shape[1])
        if cur_max > g_max:
            g_max = cur_max
        if cur_min < g_min:
            g_min = cur_min
    return g_max, g_min

def pad_and_center(img, shape):
    arr = np.zeros(shape, np.int8)
    h_offset = (shape[0] // 2) - (img.shape[0] // 2)
    w_offset = (shape[1] // 2) - (img.shape[1] // 2)
    arr[h_offset:img.shape[0]+h_offset, w_offset:img.shape[1]+w_offset] = img
    return arr
