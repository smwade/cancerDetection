import os
from os.path import join
import cv2
from glob import glob
import pathlib
import numpy as np
from PIL import Image
from tqdm import tqdm

from cancer.datasets import get_sipakmed
from cancer.variables import ABNORMAL_CELL_TYPES_SIP, NORMAL_CELL_TYPES_SIP, CANCER_DATA_DIR
from cancer.utils import read_bmp


def process_sipkamed(out_dir, width, height):
    data = get_sipakmed(cache=True)

    if not os.path.exists(join(out_dir, 'normal')):
        os.makedirs(join(out_dir, 'normal'))
    if not os.path.exists(join(out_dir, 'abnormal')):
        os.makedirs(join(out_dir, 'abnormal'))
    if not os.path.exists(join(out_dir, 'normal', 'images')):
        os.makedirs(join(out_dir, 'normal', 'images'))
    if not os.path.exists(join(out_dir, 'normal', 'masks')):
        os.makedirs(join(out_dir, 'normal', 'masks'))
    if not os.path.exists(join(out_dir, 'abnormal', 'images')):
        os.makedirs(join(out_dir, 'abnormal', 'images'))
    if not os.path.exists(join(out_dir, 'abnormal', 'masks')):
        os.makedirs(join(out_dir, 'abnormal', 'masks'))
        

    image_list = []
    mask_list = []

    for cell_type in NORMAL_CELL_TYPES_SIP:
        d = data[cell_type]
        for img_path, cyto_polys in zip(d['imgs'], d['cytos']):
            img = read_bmp(img_path)
            mask = np.zeros(img.shape[:2],dtype=np.uint8)
            for poly in cyto_polys:
                poly = np.expand_dims(poly, axis=0).astype(int)
                #cv2.fillPoly(mask, poly, 255)

            img = cv2.resize(img,(width,height))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = cv2.resize(mask,(width,height))
            
            img_name = cell_type + os.path.basename(img_path)
            img_name = img_name.replace('.bmp','.png')
            # mask_name = img_name.replace('.','_mask.')
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
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = cv2.resize(mask,(width,height))
            
            img_name = cell_type + os.path.basename(img_path)
            img_name = img_name.replace('.bmp','.png')
            #mask_name = img_name.replace('.','_mask.') TODO : for Augmentor
            mask_name = img_name
            
            cv2.imwrite(join(out_dir, 'abnormal', 'images', img_name), img)
            cv2.imwrite(join(out_dir, 'abnormal', 'masks', mask_name), mask)

            image_list.append(img)
            mask_list.append(mask)

        images = np.array(image_list)
        masks = np.array(mask_list)

        def unison_shuffled_copies(a, b):
            assert len(a) == len(b)
            p = np.random.permutation(len(a))
            return a[p], b[p]

        images, masks = unison_shuffled_copies(images, masks)

        mid = len(images) - (len(images) // 4)
        train_volume, test_volume = images[:mid], images[mid:]
        train_labels, test_labels = images[:mid], images[mid:]

        np.save(join(out_dir, 'train-volume.npy'), train_volume)
        np.save(join(out_dir, 'test-volume.npy'), test_volume)
        np.save(join(out_dir, 'train-labels.npy'), train_labels)
        np.save(join(out_dir, 'test-labels.npy'), test_labels)

        np.save(join(out_dir, 'images_arr.npy'), images)
        np.save(join(out_dir, 'masks_arr.npy'), masks)


def process_unet(out_dir, width, height, grey=False, cell_segmentor=False):
    data = get_sipakmed(cache=True)

    if grey:
        out_dir = join(out_dir, 'grey')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    if not os.path.exists(join(out_dir, 'npydata')):
        os.makedirs(join(out_dir, 'npydata'))
    if not os.path.exists(join(out_dir, 'train')):
        os.makedirs(join(out_dir, 'train'))
    if not os.path.exists(join(out_dir, 'train', 'images')):
        os.makedirs(join(out_dir, 'train', 'images'))
    if not os.path.exists(join(out_dir, 'train', 'masks')):
        os.makedirs(join(out_dir, 'train', 'masks'))
    if not os.path.exists(join(out_dir, 'test')):
        os.makedirs(join(out_dir, 'test'))
    if not os.path.exists(join(out_dir, 'test', 'images')):
        os.makedirs(join(out_dir, 'test', 'images'))

    image_list = []
    mask_list = []

    for cell_type in NORMAL_CELL_TYPES_SIP:
        for img_path, cyto_polys in zip(data[cell_type]['imgs'], data[cell_type]['cytos']):
            img = read_bmp(img_path)
            mask = np.zeros(img.shape[:2],dtype=np.uint8)
            for poly in cyto_polys:
                poly = np.expand_dims(poly, axis=0).astype(int)
                if cell_segmentor:
                    cv2.fillPoly(mask, poly, 255)

            img = cv2.resize(img,(width,height))
            mask = cv2.resize(mask,(width,height))
            if grey:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            img_name = f'{cell_type}_{os.path.basename(img_path)}'
            img_name = img_name.replace('.bmp','.png')
            
            cv2.imwrite(join(out_dir, 'train', 'images', img_name), img)
            cv2.imwrite(join(out_dir, 'train', 'masks', img_name), mask)
            image_list.append(img)
            mask_list.append(mask)

    for cell_type in ABNORMAL_CELL_TYPES_SIP:
        for img_path, cyto_polys in zip(data[cell_type]['imgs'], data[cell_type]['cytos']):
            img = read_bmp(img_path)
            mask = np.zeros(img.shape[:2],dtype=np.uint8)
            for poly in cyto_polys:
                poly = np.expand_dims(poly, axis=0).astype(int)
                cv2.fillPoly(mask, poly, 255)

            img = cv2.resize(img,(width,height))
            mask = cv2.resize(mask,(width,height))
            if grey:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            img_name = f'{cell_type}_{os.path.basename(img_path)}'
            img_name = img_name.replace('.bmp','.png')
            
            cv2.imwrite(join(out_dir, 'images', img_name), img)
            cv2.imwrite(join(out_dir, 'masks', img_name), mask)
            image_list.append(img)
            mask_list.append(mask)

    images = np.array(image_list).squeeze()
    masks = np.array(mask_list).squeeze()

    images, masks = unison_shuffled_copies(images, masks)

    mid = len(images) - (len(images) // 4)
    train_volume, test_volume = images[:mid], images[mid:]
    train_masks, test_masks = images[:mid], images[mid:]

    np.save(join(out_dir, 'npydata', 'train_images.npy'), train_volume)
    np.save(join(out_dir, 'npydata', 'test_images.npy'), test_volume)
    np.save(join(out_dir, 'npydata', 'train_masks.npy'), train_masks)
    np.save(join(out_dir, 'npydata', 'test_masks.npy'), test_masks)

    np.save(join(out_dir, 'npydata', 'images_arr.npy'), images)
    np.save(join(out_dir, 'npydata', 'masks_arr.npy'), masks)


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def process_cell_gan(out_dir):
    """
    Adds indavidual cells to their own folder
    """
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
        pathlib.Path(join(out_dir, cell_name)).mkdir(parents=True, exist_ok=True)
        for i, img_path in tqdm(enumerate(glob(data_dir))):
            img = Image.open(img_path)
            img.save(join(out_dir, cell_name, f"{i}.png",), 'PNG')

def resize_images(in_dir, out_dir, h, w):
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
    # process_unet(join(CANCER_DATA_DIR, 'SIPaKMeD', "processed_data"), 1024, 1024, grey=True)
    # process_cell_gan('/Users/seanwade/Desktop/cells')
    resize_images('/Users/seanwade/Desktop/cells', '/Users/seanwade/Desktop/cells/processed_large', 256, 256)
