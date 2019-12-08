import os
from os.path import join
import click
import random
from itertools import cycle

from mediaug.dataset import Dataset
from mediaug.augment import add_cell, randomly_insert_cells
from mediaug.image_utils import get_blank_mask


@click.command()
@click.option('--slide_dir', type=click.Path(exists=True), required=True)
@click.option('--cell_dir', type=click.Path(exists=True), required=True)
@click.option('--out_dir', type=click.Path(), required=True)
@click.option('--num', type=int, required=True)
@click.option('--max_cells', type=int, default=7)
def generate_augment_dataset(slide_dir, cell_dir, out_dir, num, max_cells):
    """ Adds cells to slides to produce a weekly supervised training
    dataset for SIPaKMeD dataset.
    """
    slides = Dataset(slide_dir)
    cells = Dataset(cell_dir)
    out_ds = Dataset(out_dir, ['all'])

    good_slides = slides['superficial-intermediate'] + slides['parabasal']
    random.shuffle(good_slides)
    slide_generator = cycle(good_slides)

    bad_cells = list(set(slides.classes) - set(['superficial-intermediate', 'parabasal']))

    for i in range(num):
        slide = next(slide_generator)
        new_img, new_mask = randomly_insert_cells(slide.img, get_blank_mask(slide.img), cells, bad_cells, (0,max_cells))
        out_ds.add_data(new_img, new_mask, 'all', i)


if __name__ == '__main__':
    generate_augment_dataset()
