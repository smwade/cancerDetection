import os
from os.path import join
import cv2
import numpy as np

from cancer.datasets import get_sipakmed
from cancer.variables import ABNORMAL_CELL_TYPES_SIP, NORMAL_CELL_TYPES_SIP
from cancer.utils.utils import read_bmp

w, h = 512, 512
BASE_DIR = '/Users/seanwade/Desktop/data'

data = get_sipakmed()

for cell_type in NORMAL_CELL_TYPES_SIP:
    d = data[cell_type]
    for img_path, cyto_polys in zip(d['imgs'], d['cytos']):
        img = read_bmp(img_path)
        mask = np.zeros(img.shape[:2],dtype=np.uint8)
        for poly in cyto_polys:
            poly = np.expand_dims(poly, axis=0).astype(int)
            #cv2.fillPoly(mask, poly, 255)


        img = cv2.resize(img,(w,h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask,(w,h))
        
        img_name = cell_type + os.path.basename(img_path)
        img_name = img_name.replace('.bmp','.png')
        mask_name = img_name.replace('.','_mask.')
        
        cv2.imwrite(join(BASE_DIR, 'normal', 'images', img_name), img)
        cv2.imwrite(join(BASE_DIR, 'normal', 'masks', mask_name), mask)

for cell_type in ABNORMAL_CELL_TYPES_SIP:
    d = data[cell_type]
    for img_path, cyto_polys in zip(d['imgs'], d['cytos']):
        img = read_bmp(img_path)
        mask = np.zeros(img.shape[:2],dtype=np.uint8)
        for poly in cyto_polys:
            poly = np.expand_dims(poly, axis=0).astype(int)
            cv2.fillPoly(mask, poly, 255)


        img = cv2.resize(img,(w,h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask,(w,h))
        
        img_name = cell_type + os.path.basename(img_path)
        img_name = img_name.replace('.bmp','.png')
        mask_name = img_name.replace('.','_mask.')
        
        cv2.imwrite(join(BASE_DIR, 'abnormal', 'images', img_name), img)
        cv2.imwrite(join(BASE_DIR, 'abnormal', 'masks', mask_name), mask)