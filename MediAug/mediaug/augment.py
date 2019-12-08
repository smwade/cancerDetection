import Augmentor
import numpy as np
from Augmentor.Operations import Operation
import os
from os.path import join
from random import randint
import random
import cv2

from skimage import exposure
from skimage.exposure import match_histograms

from mediaug.image_utils import is_greyscale, rotate, soften_mask, image_on_image_alpha

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


# def pick_random_cell(ds):
#     cell_type = random.choice(ABNORMAL_CELL_TYPES_SIP + NORMAL_CELL_TYPES_SIP)
#     indx = randint(0, len(ds[cell_type]['imgs']))
#     poly_list = ds[cell_type]['cytos'][indx] 
#     return ds[cell_type]['imgs'][indx], random.choice(poly_list)


# def randomly_insert_cells(img):
#     w, h = img.shape[:2]
#     num_cells_to_insert = randint(0,3)
#     ds = get_sipakmed(cache=True)
#     for i in range(num_cells_to_insert):
#         cell_path, cell_poly = pick_random_cell(ds)
#         cell_img = read_png(cell_path)
#         offset = (randint(0, w), randint(0, h))
#         angle = randint(0,360)
#         img = add_cell(img, cell_img, cell_poly, offset, angle, 0)
#     return img


def add_cell(bg, bg_mask, fg, orig_fg_mask, pos, angle=0, scale=1,
                blend_method=None, blend_edge_amount=1):
    """ adds a cell to base image
    Args:
      bg (np.array): background img
      bg_mask (np.array): background img mask
      fg (np.array): foreground img to add
      fg_mask (np.array): foreground img mask
      pos (x,y): where to put the fg
      angle (float): angle of fg in degrees
      b (int): the abount to blend from the mask

    Returns:
      img (np.array)
      mask (np.array)
    """
    fg = cv2.resize(fg, (0,0), fx=scale, fy=scale)
    orig_fg_mask = cv2.resize(orig_fg_mask, (0,0), fx=scale, fy=scale) 

    fg_mask = cv2.cvtColor(orig_fg_mask, cv2.COLOR_BGR2GRAY)

    # blend mode
    if blend_method == 'hist':
        fg = match_histograms(fg, bg, multichannel=True)

    # prep the mask : TODO: improve this
    fg_mask[fg_mask == 51] = 160
    fg_mask[fg_mask == 170] = 250

    fg = rotate(fg, angle)
    fg_mask = rotate(fg_mask, angle)
    fg_mask = soften_mask(fg_mask, amount=blend_edge_amount) # FIXME: this also blurs nucleus

    new_img = image_on_image_alpha(bg, fg, fg_mask, pos)
    fg_mask[fg_mask != 0] = 255
    new_mask = image_on_image_alpha(bg_mask, orig_fg_mask, fg_mask, pos)
    return new_img, new_mask


def cell_augment(ds, outdir):
    pass
