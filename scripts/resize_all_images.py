import os
from os.path import join
from PIL import Image
import pathlib
import click

from cancer.utils import create_dirs


def is_image(path):
    return path.endswith('.png') or path.endswith('.jpg') or path.endswith('.jpeg') 

@click.command()
@click.option('-i', '--input', type=click.Path(exists=True), required=True)
@click.option('-o', '--output', type=click.Path(), required=True)
@click.option('-w', '--width', type=int, required=True)
@click.option('-h', '--height', type=int, required=True)
def resize_images(in_dir, out_dir, h, w):
    """resize all images in a dir"""
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    for subdir, dirs, files in os.walk(in_dir):
        create_dirs(subdir, dirs)
        for file in files:
            if is_image(file):
                im = Image.open(join(subdir, file))
                imResize = im.resize((h, w), Image.ANTIALIAS)
                cell_type = os.path.basename(subdir)
                pathlib.Path(join(out_dir, cell_type)).mkdir(parents=True, exist_ok=True)
                imResize.save(join(out_dir, cell_type, file), 'PNG')

if __name__ == '__main__':
    resize_images()
