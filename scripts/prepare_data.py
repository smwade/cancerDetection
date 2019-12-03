import os
import shutil
from os.path import join
import cv2
from glob import glob
import pathlib
import numpy as np
from PIL import Image
from tqdm import tqdm

from cancer.datasets import get_sipakmed
from cancer.variables import ABNORMAL_CELL_TYPES_SIP, NORMAL_CELL_TYPES_SIP, CANCER_DATA_DIR
from cancer.utils import read_bmp, unison_shuffled_copies

# helper functions
def create_dirs(base_path, dir_list):
    for new_dir in dir_list:
        pathlib.Path(join(base_path, new_dir)).mkdir(parents=True)


def prepare_full_slide_sip(width=1024, height=1024):
    out_dir = join(CANCER_DATA_DIR, 'SIPaKMeD', 'processed_data', 'full_slide_classification')
    create_dirs(out_dir, ['normal/images', 'abnormal/images', 'normal/masks', 'abnormal/masks', 'npydata'])

    data = get_sipakmed(cache=False)
    image_list = []
    mask_list = []

    for cell_type in NORMAL_CELL_TYPES_SIP:
        d = data[cell_type]
        for img_path, cyto_polys in zip(d['imgs'], d['cytos']):
            img = read_bmp(img_path)
            mask = np.zeros(img.shape[:2],dtype=np.uint8)
            for poly in cyto_polys:
                poly = np.expand_dims(poly, axis=0).astype(int)
                cv2.fillPoly(mask, poly, 255)

            img = cv2.resize(img,(width,height))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = cv2.resize(mask,(width,height))
            
            img_name = cell_type + os.path.basename(img_path)
            img_name = img_name.replace('.bmp','.png')
            mask_name = img_name
            
            cv2.imwrite(join(out_dir, 'normal', 'images', img_name), img)
            cv2.imwrite(join(out_dir, 'normal', 'masks', mask_name), mask)
            image_list.append(img)
            mask_list.append(mask)

    for cell_type in ABNORMAL_CELL_TYPES_SIP:
        d = data[cell_type]
        for img_path, cyto_polys in zip(d['imgs'], d['cytos']):
            img = read_bmp(img_path)
            mask = np.zeros(img.shape[:2],dtype=np.uint8)
            for poly in cyto_polys:
                poly = np.expand_dims(poly, axis=0).astype(int)
                cv2.fillPoly(mask, poly, 255)


            img = cv2.resize(img,(width,height))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = cv2.resize(mask,(width,height))
            
            img_name = cell_type + os.path.basename(img_path)
            img_name = img_name.replace('.bmp','.png')
            mask_name = img_name
            
            cv2.imwrite(join(out_dir, 'abnormal', 'images', img_name), img)
            cv2.imwrite(join(out_dir, 'abnormal', 'masks', mask_name), mask)

            image_list.append(img)
            mask_list.append(mask)

        images = np.array(image_list)
        masks = np.array(mask_list)

        images, masks = unison_shuffled_copies(images, masks)

        mid = len(images) - (len(images) // 4)
        train_volume, test_volume = images[:mid], images[mid:]
        train_labels, test_labels = images[:mid], images[mid:]

        np.save(join(out_dir, 'npydata', 'train_images.npy'), train_volume)
        np.save(join(out_dir, 'npydata', 'test_images.npy'), test_volume)
        np.save(join(out_dir, 'npydata', 'train_masks.npy'), train_labels)
        np.save(join(out_dir, 'npydata', 'test_masks.npy'), test_labels)
        np.save(join(out_dir, 'npydata', 'images_arr.npy'), images)
        np.save(join(out_dir, 'npydata', 'masks_arr.npy'), masks)


def prepare_indavidual_cell_sip():
    """
    Adds indavidual cells to their own folder
    """
    out_dir = join(CANCER_DATA_DIR, 'SIPaKMeD', 'processed_data', 'indavidual_cells')
    cell_types = [
        'im_Metaplastic',
        'im_Dyskeratotic',
        'im_Superficial-Intermediate',
        'im_Parabasal',
        'im_Koilocytotic'
    ]
    for cell_name in cell_types:
        data_dir = join(CANCER_DATA_DIR, 'SIPaKMeD', cell_name, 'CROPPED', '*.bmp')
        cell_name = cell_name.replace('im_','').lower()
        create_dirs(out_dir, [cell_name])
        for i, img_path in tqdm(enumerate(glob(data_dir))):
            img = Image.open(img_path)
            img = img.resize((160,300),Image.ANTIALIAS)
            img.save(join(out_dir, cell_name, f"{i}.png",), 'PNG')


def resize_images(in_dir, out_dir, h, w):
    """resize all images in a dir"""
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    for subdir, dirs, files in os.walk(in_dir):
        for file in files:
            if file == '.DS_Store':
                continue
            im = Image.open(join(subdir, file))
            imResize = im.resize((h, w), Image.ANTIALIAS)
            cell_type = os.path.basename(subdir)
            pathlib.Path(join(out_dir, cell_type)).mkdir(parents=True, exist_ok=True)
            imResize.save(join(out_dir, cell_type, file), 'PNG')


if __name__ == '__main__':
    prepare_full_slide_sip()
    prepare_indavidual_cell_sip()
