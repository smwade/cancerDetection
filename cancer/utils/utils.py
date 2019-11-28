import os
import cv2
from os.path import join
from collections import defaultdict
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

from matplotlib import pyplot as plt


def read_bmp(img_path):
    cv2.imread(img_path)

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