import Augmentor
from Augmentor.Operations import Operation
import os
from os.path import join
from random import randint
import random

from cancer.variables import CANCER_DATA_DIR, ABNORMAL_CELL_TYPES_SIP, NORMAL_CELL_TYPES_SIP
from cancer.datasets import get_sipakmed
from cancer.utils import add_cell, read_png

CELL_TYPES_SIP = ABNORMAL_CELL_TYPES_SIP + NORMAL_CELL_TYPES_SIP


def get_data_generator(image_path, mask_path, batch_size=1):
    pipeline = Augmentor.Pipeline(image_path)
    if mask_path is not None:
        pipeline.ground_truth(mask_path)

    pipeline.rotate(probability=0.5, max_left_rotation=25, max_right_rotation=25)
    pipeline.flip_left_right(probability=0.5)
    pipeline.zoom_random(probability=0.5, percentage_area=0.6)
    pipeline.flip_top_bottom(probability=0.5)
    pipeline.random_distortion(probability=.3, grid_width=8, grid_height=8, magnitude=5)
    pipeline.crop_random(.05, .85)
    
    gen = pipeline.keras_generator(batch_size=batch_size)

    return gen

def pick_random_cell(ds):
    cell_type = random.choice(CELL_TYPES_SIP)
    indx = randint(0, len(ds[cell_type]['imgs']))
    poly_list = ds[cell_type]['cytos'][indx] 
    return ds[cell_type]['imgs'][indx], random.choice(poly_list)

def randomly_insert_cells(img):
    w, h = img.shape[:2]
    num_cells_to_insert = randint(0,3)
    ds = get_sipakmed(cache=True)
    for i in range(num_cells_to_insert):
        cell_path, cell_poly = pick_random_cell(ds)
        cell_img = read_png(cell_path)
        offset = (randint(0, w), randint(0, h))
        angle = randint(0,360)
        img = add_cell(img, cell_img, cell_poly, offset, angle, 0)
    return img
