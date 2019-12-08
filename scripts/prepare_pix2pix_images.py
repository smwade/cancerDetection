import os
from os.path import join
from math import floor
import click

from mediaug.data_prep import make_pix2pix_format
from mediaug.image_utils import save_img, read_png
from mediaug.utils import create_dirs


@click.command()
@click.option('-i', '--image_dir', type=click.Path(exists=True), required=True)
@click.option('-m', '--mask_dir', type=click.Path(exists=True), required=True)
@click.option('-o', '--out_dir', type=click.Path(), required=True)
@click.option('-r', '--split_ratio', type=float, required=True)
def prepare_pix2pix_images(image_dir, mask_dir, out_dir, split_ratio):
    """ Prepares images to be in correct format for Pix2Pix algorithm."""
    images_list = os.listdir(image_dir)
    create_dirs(out_dir, ['train', 'val', 'test', 'all'])
    
    # calc split cutoffs
    train_cutoff = floor(split_ratio * len(images_list))
    val_cutoff = train_cutoff + floor((len(images_list)-train_cutoff) // 2)

    for i, image_path in enumerate(images_list):
        mask_path = join(mask_dir, os.path.basename(image_path))
        img = read_png(join(image_dir, image_path))
        mask = read_png(join(mask_dir, mask_path))
        new_img = make_pix2pix_format(img, mask)

        # split the data
        if i < train_cutoff:
            save_img(new_img, join(out_dir, 'train', f'{i}.jpg'))
        elif i < val_cutoff:
            save_img(new_img, join(out_dir, 'val', f'{i}.jpg'))
        else:
            save_img(new_img, join(out_dir, 'test', f'{i}.jpg'))
        save_img(new_img, join(out_dir, 'all', f'{i}.jpg'))


if __name__ == '__main__':
    prepare_pix2pix()
