import Augmentor
from Augmentor.Operations import Operation
import os
from os.path import join

from cancer.variables import BASE_DATA_DIR


def get_data_generator(image_path, mask_path, batch_size=1):
    pipeline = Augmentor.Pipeline(image_path)
    pipeline.ground_truth(mask_path)

    pipeline.rotate(probability=0.5, max_left_rotation=25, max_right_rotation=25)
    pipeline.flip_left_right(probability=0.5)
    pipeline.zoom_random(probability=0.5, percentage_area=0.6)
    pipeline.flip_top_bottom(probability=0.5)
    pipeline.random_distortion(probability=.3, grid_width=8, grid_height=8, magnitude=5)
    pipeline.crop_random(.05, .85)

    return pipeline.keras_generator(batch_size=batch_size, scaled=False)


if __name__ == '__main__':
    data_path = join(BASE_DATA_DIR, 'SIPaKMeD', 'processed_data', 'test', 'images')
    mask_path = join(BASE_DATA_DIR, 'SIPaKMeD', 'processed_data', 'test', 'masks')
    gen = get_data_generator(data_path, mask_path, batch_size=1)
