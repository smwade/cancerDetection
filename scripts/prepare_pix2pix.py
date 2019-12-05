import os
from os.path import join
from math import floor
import cv2
import click

from cancer.visulize import show_image
from cancer.utils import read_png, create_dirs
from cancer.data_prep import make_pix2pix_format


@click.command()
@click.option('-i', '--input_dir', type=click.Path(exists=True), required=True)
@click.option('-o', '--out_dir', type=click.Path(), required=True)
def prepare_pix2pix(input_dir, out_dir, train_split_ratio):
    images_list = os.listdir(join(input_dir, 'images'))
    create_dirs(out_dir, ['train', 'val', 'test')
    
    # calc split cutoffs
    train_cutoff = floor(train_split_ratio * len(images_list))
    val_cutoff = train_cutoff + floor((len(images_list)-train_cutoff) // 2)

    for i, image_path in enumerate(images_list):
        mask_path = join(input_dir, 'masks', os.path.basename(image_path))
        img = read_png(image_path)
        mask = read_png(mask_path)
        new_img = make_pix2pix_format(img, mask)

        # split the data
        if i < train_cutoff:
            cv2.imwrite(join(out_dir, 'train', f'{i}.jpg'), new_img)
        elif i < val_cutoff:
            cv2.imwrite(join(out_dir, 'val', f'{i}.jpg'), new_img)
        else:
            cv2.imwrite(join(out_dir, 'test', f'{i}.jpg'), new_img)

if __name__ == '__main__':
    prepare_pix2pix()
