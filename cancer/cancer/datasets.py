import os
from os.path import join
import pickle
import cv2
from glob import glob
import numpy as np
from collections import defaultdict

from cancer.utils import read_dat_file, read_png, read_bmp
from cancer.variables import CANCER_DATA_DIR, NORMAL_CELL_TYPES_SMEAR, ABNORMAL_CELL_TYPES_SMEAR


def get_cell_type_dataset(use_cached=True):
    """
    returns data as 2 arrays.
    images: [n,256,256,3]
    lables: [n, 5]
    """
    data_dir = join(CANCER_DATA_DIR, 'SIPaKMeD', 'processed_data', 'cwl', 'large')
    if use_cached:
        with open(join(data_dir, 'data.pkl'), 'rb') as f:
            return pickle.load(f)
    cell_types = [
        'metaplastic',
        'dyskeratotic',
        'superficial-Intermediate',
        'parabasal',
        'koilocytotic'
    ]
    images, labels = [], []
    for i, cell_name in enumerate(cell_types):
        for img_path in os.listdir(join(data_dir, cell_name)):
            img = read_png(join(data_dir, cell_name, img_path))
            label = np.zeros(5)
            label[i] = 1
            images.append(img)
            labels.append(label)
    images, labels = np.array(images).squeeze(), np.array(labels).squeeze()
    with open(join(data_dir, 'data.pkl'), 'wb') as f:
        pickle.dump((images, labels), f)
    return images, labels


def get_smear_dataset(paths=True, flat=False):
    data = {}
    DATA_DIR = join(CANCER_DATA_DIR, 'SMEAR2005', 'pictures')
    for folder in os.listdir(DATA_DIR):
        if folder == '.DS_Store':
            continue
        iamge_list = []
        mask_list = []
        for mask_path in glob(join(DATA_DIR, folder, '*-d.bmp')):
            basename = os.path.basename(mask_path)
            img_path = join(DATA_DIR, folder, basename.replace('-d',''))
            iamge_list.append(img_path)
            mask_list.append(mask_path)
        data[folder] = {
            'imgs': iamge_list,
            'masks': mask_list
        }
    if not flat:
        return data
    
    imgs, masks, labels = [], [], []
    for val in ABNORMAL_CELL_TYPES_SMEAR:
        for img_path, mask_path in zip(data[val]['imgs'], data[val]['masks']):
            imgs.append(read_bmp(img_path))
            masks.append(read_bmp(mask_path))
            labels.append(1)
    for val in NORMAL_CELL_TYPES_SMEAR:
        for img_path, mask_path in zip(data[val]['imgs'], data[val]['masks']):
            imgs.append(read_bmp(img_path))
            masks.append(read_bmp(mask_path))
            labels.append(0)
    return np.stack(imgs), np.stack(masks), np.array(labels).reshape((-1,1))
            

def get_sipakmed(cache=True):
    # Read the SIPaKMeD dataset
    if cache:
        with open(join(CANCER_DATA_DIR, 'SIPaKMeD', 'sipakmed.pkl'), 'rb') as f:
            return pickle.load(f)
    dataset = {}
    cell_types = [
        'im_Metaplastic',
        'im_Dyskeratotic',
        'im_Superficial-Intermediate',
        'im_Parabasal',
        'im_Koilocytotic'
    ]

    for cell_name in cell_types:
        DATA_DIR = join(CANCER_DATA_DIR, 'SIPaKMeD', cell_name)
        imgs = []
        cytos = []
        nucli = []
        slide_num = 1
        while True:
            key = '{:03d}'.format(slide_num)
            img_file_name = f'{key}.bmp'
            if img_file_name not in set(os.listdir(DATA_DIR)):
                break

            imgs.append(join(DATA_DIR, img_file_name))

            cyt_list = []
            nuc_list = []
            cell_num = 1
            while True:
                cyt_file_name = f'{key}_cyt{cell_num:02d}.dat'
                nuc_file_name = f'{key}_nuc{cell_num:02d}.dat'
                if cyt_file_name not in set(os.listdir(DATA_DIR)):
                    break
                cyt_list.append(read_dat_file(join(DATA_DIR, cyt_file_name)))
                nuc_list.append(read_dat_file(join(DATA_DIR, nuc_file_name)))
                cell_num += 1

            cytos.append(cyt_list)
            nucli.append(nuc_list)
                
            slide_num += 1
            
        cell_name = cell_name.replace('im_','').lower()
        dataset[cell_name] = {
            'imgs': imgs,
            'cytos': cytos,
            'nucli': nucli
        }
        with open(join(CANCER_DATA_DIR, 'SIPaKMeD', 'sipakmed.pkl'), 'wb') as f:
            pickle.dump(dataset, f)

    return dataset
