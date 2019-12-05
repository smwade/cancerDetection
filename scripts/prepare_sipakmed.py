import os
from os.path import join
from glob import glob
from tqdm import tqdm
import click

from cancer.utils import read_bmp, read_dat_file, save_img, \
    generate_full_mask, generate_cell_mask, create_dirs


@click.command()
@click.option('-i', '--input_dir', type=click.Path(exists=True), required=True)
@click.option('-o', '--out_dir', type=click.Path(), required=True)
def prepare_sipakmed(input_dir, out_dir):
    """
    Prepare the data from the SIPKaMeD dataset. Puts in the form:
    slides/
        metaplastic/
            images/
            masks
        ...
    cells/
        metaplastic/
            image/
            mask/
    """
    cell_types = [
        'im_Metaplastic',
        'im_Dyskeratotic',
        'im_Superficial-Intermediate',
        'im_Parabasal',
        'im_Koilocytotic'
    ]

    new_cell_types = [x.replace('im_','').lower() for x in cell_types]
    create_dirs(out_dir, [f'slides/{x}' for x in new_cell_types])
    
    # save full slides
    print('Adding full slides...')
    for full_cell_name, cell_name in zip(cell_types, new_cell_types):
        cur_dir = join(input_dir, full_cell_name)
        create_dirs(join(out_dir, 'slides', cell_name), ['images', 'masks'])
        img_path_list = glob(join(cur_dir, '*.bmp'))
        for img_path in tqdm(img_path_list):
            img = read_bmp(img_path)
            img_num = os.path.basename(img_path).split('.')[0]

            cyto_path_list = glob(join(cur_dir, f'{img_num}_cyt*'))
            nuc_path_list = glob(join(cur_dir, f'{img_num}_nuc*'))

            cyto_list = [read_dat_file(x) for x in cyto_path_list]
            nuc_list = [read_dat_file(x) for x in nuc_path_list]

            mask = generate_full_mask(img, cyto_list, nuc_list)

            save_img(join(out_dir, 'slides', cell_name, 'images', f'{cell_name}_{img_num}.png'), img)
            save_img(join(out_dir, 'slides', cell_name, 'masks', f'{cell_name}_{img_num}.png'), mask)

    # save indavidual cells 
    print('Adding indavidual cells...')
    create_dirs(out_dir, [f'cells/{x}' for x in new_cell_types])
    for full_cell_name, cell_name in zip(cell_types, new_cell_types):
        cur_dir = join(input_dir, full_cell_name, 'CROPPED')
        create_dirs(join(out_dir, 'cells', cell_name), ['image', 'mask'])
        for img_num, img_path in tqdm(enumerate(glob(join(cur_dir, '*.bmp')))):
            img = read_bmp(img_path)
            name = os.path.basename(img_path).split('.')[0]

            cyto_path = join(cur_dir, f'{name}_cyt.dat')
            nuc_path = join(cur_dir, f'{name}_nuc.dat')

            cyto = read_dat_file(cyto_path)
            nuc = read_dat_file(nuc_path)

            mask = generate_cell_mask(img, cyto, nuc)

            save_img(join(out_dir, 'cells', cell_name, 'image', f'{cell_name}_{img_num}.png'), img)
            save_img(join(out_dir, 'cells', cell_name, 'mask', f'{cell_name}_{img_num}.png'), mask)


if __name__ == '__main__':
    prepare_sipakmed()
