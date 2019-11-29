import os
from os.path import join
import cv2
import numpy as np

from cancer.datasets import get_sipakmed
from cancer.variables import ABNORMAL_CELL_TYPES_SIP, NORMAL_CELL_TYPES_SIP, BASE_DATA_DIR
from cancer.utils.utils import read_bmp


def process_sipkamed(out_dir, width, height):
    data = get_sipakmed(cache=True)

    image_list = []
    mask_list = []

    # for cell_type in NORMAL_CELL_TYPES_SIP:
    #     d = data[cell_type]
    #     for img_path, cyto_polys in zip(d['imgs'], d['cytos']):
    #         img = read_bmp(img_path)
    #         mask = np.zeros(img.shape[:2],dtype=np.uint8)
    #         for poly in cyto_polys:
    #             poly = np.expand_dims(poly, axis=0).astype(int)
    #             #cv2.fillPoly(mask, poly, 255)

    #         img = cv2.resize(img,(width,height))
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         mask = cv2.resize(mask,(width,height))
            
    #         img_name = cell_type + os.path.basename(img_path)
    #         img_name = img_name.replace('.bmp','.png')
    #         mask_name = img_name.replace('.','_mask.')
            
    #         cv2.imwrite(join(out_dir, 'normal', 'images', img_name), img)
    #         cv2.imwrite(join(out_dir, 'normal', 'masks', mask_name), mask)
    #         image_list.append(img)
    #         mask_list.append(mask)

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
            mask_name = img_name.replace('.','_mask.')
            
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


if __name__ == '__main__':
    process_sipkamed(join(BASE_DATA_DIR, 'SIPaKMeD', "processed_data"), 512, 512)
