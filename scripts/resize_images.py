import pathlib
import os
from os.path import join
from PIL import Image


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
    in_dir = '/home/seanwade/sean/data/SIPaKMeD/processed_data/full_slide_classification/classes'
    out_dir = '/home/seanwade/sean/data/SIPaKMeD/processed_data/full_slide_classificationclasses_small'
    resize_images(in_dir, out_dir, 512, 512)
