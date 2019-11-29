import Augmentor
from Augmentor.Operations import Operation
import os
from os.path import join

from cancer.variables import BASE_DATA_DIR

# -- OPERATIONS ----
class Test(Operation):
    pass

data_path = join(BASE_DATA_DIR, 'SIPaKMeD', 'processed_data', 'test', 'images')
mask_path = join(BASE_DATA_DIR, 'SIPaKMeD', 'processed_data', 'test', 'masks')

pipeline = Augmentor.Pipeline(data_path)
pipeline.ground_truth(mask_path)

pipeline.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
pipeline.flip_left_right(probability=0.5)
pipeline.zoom_random(probability=0.5, percentage_area=0.8)
pipeline.flip_top_bottom(probability=0.5)
#pipeline.random_distortion

pipeline.sample(100)