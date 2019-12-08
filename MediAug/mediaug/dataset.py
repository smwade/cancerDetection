import os
from os.path import join
import numpy as np
from tqdm import tqdm
from mediaug.image_utils import read_png, save_img
from mediaug.download import get_data_cache


class DataPoint:

    def __init__(self, img_path, mask_path, _class, _id=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self._class = _class
        if _id is None:
            self.id = img_path.split('.')[0]

    @property
    def img(self):
        return read_png(self.img_path)

    @property
    def mask(self):
        return read_png(self.mask_path)

    def __repr__(self):
        return f'<img_path: {self.img_path}>\n<mask_path: {self.mask_path}>'


class Dataset:

    def __init__(self, data_path=None, classes=None):
        self.data_path = data_path
        if not os.path.exists(data_path) and classes is not None:
            self._create_empty_dataset(classes)
        if not os.path.exists(data_path) and classes is None:
            raise ValueError('No data in path or classes.')
        self._parse(data_path)
    
    def _parse(self, data_path):
        self.data = {}
        categories =  [x for x in os.listdir(data_path) if not x.startswith('.')]
        self.data = {key:[] for key in categories}
        for c in categories:
            cur_dir = join(data_path, c)
            for base_name in os.listdir(join(cur_dir, 'image')):
                name = base_name.split('.')[0]
                dp = DataPoint(join(cur_dir, 'image', base_name),
                                join(cur_dir, 'mask', base_name), c, name)
                self.data[c].append(dp)

    def _create_empty_dataset(self, classes):
        os.mkdir(self.data_path)
        self.data = {key:[] for key in classes}
        for _class in classes:
            os.mkdir(join(self.data_path, _class))
            os.mkdir(join(self.data_path, _class, 'image'))
            os.mkdir(join(self.data_path, _class, 'mask'))


    def add_datapoint(self, dp):
        self.data[dp._class].append(dp)


    def add_data(self, img, mask, _class, name):
        img_path = save_img(img, join(self.data_path, _class, 'image', f'{name}.png'))
        mask_path = save_img(mask, join(self.data_path, _class, 'mask', f'{name}.png'))
        self.data[_class].append(DataPoint(img_path, mask_path, _class))


    def get_data(self, _id):
        """ Gets a datapoint by id """
        raise NotImplementedError 


    def get_metrix_keras_form(self, num_samples=-1):
        """ This is of the form:
        (x_train, y_train), (x_test, y_test)
        ex: (num_samples, 32, 32, 3)
        (num_samples, 1)
        """
        images = []
        masks = []
        for c in tqdm(self.classes):
            for dp in tqdm(self.data[c][:num_samples]):
                images.append(dp.img)
                masks.append(dp.img)
        return np.array(images), np.array(masks)

    @property
    def classes(self):
        return list(self.data.keys())

    @property
    def size(self):
        size = 0
        for c in self.classes:
            size += len(self.data[c])
        return size

    def __getitem__(self, arg):
        return self.data[arg]