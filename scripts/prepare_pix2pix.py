import os
from os.path import join
import numpy as np
import pathlib
import cv2
from PIL import Image

from matplotlib import pyplot as plt

from cancer.variables import CANCER_DATA_DIR
from cancer.utils import read_bmp, read_dat_file
from cancer.visulize import show_image
from cancer.utils import convert_array_to_poly
from PIL import ImageDraw


def create_dirs(base_path, dir_list=[]):
    pathlib.Path(base_path).mkdir(parents=True, exist_ok=True)
    for new_dir in dir_list:
        pathlib.Path(join(base_path, new_dir)).mkdir(parents=True)

data_dir = join(CANCER_DATA_DIR, 'SIPaKMeD')
out_dir = join(CANCER_DATA_DIR, 'SIPaKMeD', 'processed_data', 'pix2pix_data')

create_dirs(out_dir, ['train', 'test', 'val'])
cell_types = [
    'im_Metaplastic',
    'im_Dyskeratotic',
    'im_Superficial-Intermediate',
    'im_Parabasal',
    'im_Koilocytotic'
]

def generate_mask(img, cyto, nuc):
    w, h, _= img.shape
    mask = Image.new(mode="RGB", size=(h,w))
    ImageDraw.Draw(mask).polygon(convert_array_to_poly(cyto), outline=None, fill='#333333')
    ImageDraw.Draw(mask).polygon(convert_array_to_poly(nuc), outline=None, fill='#AAAAAA')
    return np.array(mask)

def make_pix2pix_format(img, mask):    
    img = cv2.resize(img, (256, 256))
    mask = cv2.resize(mask, (256, 256))

    h, w = 256, 512
    new_img = np.zeros((h, w, 3), np.uint8)
    new_img[:,0:w//2] = mask
    new_img[:,w//2:] = img
 
    return new_img


count = 1
for cell_name in cell_types:
    cropped_path = join(CANCER_DATA_DIR, 'SIPaKMeD', cell_name, 'CROPPED')
    file_list = sorted(os.listdir(cropped_path))
    for i in range(len(file_list)//3):
        img, cyto, nuc = file_list[(i*3)], file_list[(i*3)+1], file_list[(i*3)+2]
        img = read_bmp(join(cropped_path, img))
        cyto = read_dat_file(join(cropped_path, cyto))
        nuc = read_dat_file(join(cropped_path, nuc))
        mask = generate_mask(img, cyto, nuc)
        new_img = make_pix2pix_format(img, mask)
        
        cv2.imwrite(join(out_dir, 'train', f'{count}.jpg'), new_img)
        cv2.imwrite(join(out_dir, 'test', f'{count}.jpg'), new_img)
        cv2.imwrite(join(out_dir, 'val', f'{count}.jpg'), new_img)
        count += 1