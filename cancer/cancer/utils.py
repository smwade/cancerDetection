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

def read_png(img_path):
    return cv2.imread(img_path)

def read_dat_file(path):
    return np.loadtxt(path, delimiter=',')

def convert_array_to_poly(arr):
    return arr.flatten().tolist()

def generate_mask_from_poly(img, poly):
    w, h, _= img.shape
    mask = Image.new('L', (w, h), 0)
    ImageDraw.Draw(mask).polygon(convert_array_to_poly(poly), outline=1, fill='#ffffff')
    return np.array(mask)

def place_img_on_img(bg, fg, offset):
    w, h = fg.shape[:2]
    bg[offset[0]:offset[0]+w, offset[1]:offset[1]+h] = fg
    return bg

def crop_cell(img, poly, b=0):
    im = Image.fromarray(img)
    left = max(min(poly[:,0])-b, 0)
    right = min(max(poly[:,0])+b, img.shape[0])
    top = max(min(poly[:,1])-b, 0)
    bottom = min(max(poly[:,1])+b, img.shape[1])
    im = im.crop((left, top, right, bottom))
    return np.array(im)

def image_on_image_alpha(bg, fg, fg_mask, offset):
    mask = np.zeros(bg.shape[:2], dtype=np.uint8)
    mask = place_img_on_img(mask, fg_mask, offset)
    cell_img = place_img_on_img(bg.copy(), fg, offset)

    alpha = mask.astype(float)/255
    alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)

    fg = cv2.multiply(alpha, cell_img.astype(float))
    bg = cv2.multiply(1.0 - alpha, bg.astype(float))

    return cv2.add(fg, bg).astype(np.uint8)

def add_cell(img, cell_img, cell_poly, offset, angle, b):
    cropped_cell = crop_cell(cell_img, cell_poly, b=b)
    cell_mask = generate_mask_from_poly(cell_img, cell_poly)
    cropped_cell_mask = crop_cell(cell_mask, cell_poly, b=b)
    cropped_cell = rotate(cropped_cell, angle)
    cropped_cell_mask = rotate(cropped_cell_mask, angle)
    mask = soften_mask(cropped_cell_mask)
    return image_on_image_alpha(img, cropped_cell, mask, offset)

def soften_mask(mask, amount=5):
    kernel = np.ones((5,5), np.uint8) 
    mask_dilation = cv2.dilate(mask, kernel, iterations=amount)
    blur = cv2.GaussianBlur(mask_dilation,(21,21),0)
    return cv2.max(mask, blur)

def image_center(image):
    (h, w) = image.shape[:2]
    return (w // 2, h // 2)

def rotate(image, angle):
    """
    rotates an image by angle, increases the dimension as necessary
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

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

def convert_tiff(path, out_dir):
    from PIL import Image, ImageSequence
    import numpy as np

    im = Image.open(path)
    slices = [np.array(i) for i in ImageSequence.Iterator(im)]
    for i, s in enumerate(slices):
        img = Image.fromarray(s)
        img.save(os.path.join(out_dir, f'out-{i}.png'))

def create_mask(cyto_poly, nuc_poly, w, h):
    """Create a color mask for pix2pix"""
    raise NotImplementedError

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
