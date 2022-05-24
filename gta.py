""" camvid.py
    This script is to convert CamVid dataset to tfrecord format.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

from utils import make_dirs
from tfrecord import *

from poly import Segment, ImageSegments, Polygon, Line, Point, SegmentsDebugger, Shape, gen_next_index, \
    generate_first_line, generate_line, is_inverting_line, derive_image_segments, derive_masks

H = 692
W = 1283
class_names = np.array([
    'bg', 'yl', 'wl'
])

cmap = np.array([
    [64, 64, 64],
    [128, 128, 0],
    [256, 256, 256]]
)

cb = np.array([
    0.2595,
    0.1826,
    4.5640,
    0.1417,
    0.5051,
    0.3826,
    9.6446,
    1.8418,
    6.6823,
    6.2478,
    3.0,
    7.3614])

label_info = {
    'name': class_names,
    'num_class': len(class_names),
    'id': np.arange(len(class_names)),
    'cmap': cmap,
    'cb': cb
}

json_class_to_class_id = {
    "line": 1,
    "w_line": 2
}

class_names = {
    0: 'bg', 1: 'yl', 2: 'wl'
}


def parse(line, root):
    line = line.rstrip()
    line = line.replace('/SegNet/CamVid', root)
    return line.split(' ')


def load_path(txt_path, root):
    with open(txt_path) as f:
        img_gt_pairs = [parse(line, root) for line in f.readlines()]
    return img_gt_pairs


def load_splited_path(txt_path, root):
    images = []
    labels = []
    with open(txt_path) as f:
        for line in f.readlines():
            image_path, label_path = parse(line, root)
            images.append(image_path)
            labels.append(label_path)
    return images, labels


class ImageInfo:
    def __init__(self, path=None, json=None, indir=None, label=None):
        self.__json = json
        self.__path = path
        self.__label = label
        self.indir = indir

    @property
    def json(self):
        return os.path.join(self.indir, self.__json)

    @property
    def path(self):
        return os.path.join(self.indir, self.__path)

    @property
    def label(self):
        return os.path.join(self.indir, self.__label)

    @label.setter
    def label(self, val):
        self.__label = val

    @json.setter
    def json(self, val):
        self.__json = val

    @path.setter
    def path(self, val):
        self.__path = val


def deduce_image_infos(indir, ext="jpg", json_ext="labels.json", label_ext="label"):
    file_names = os.listdir()

    image_infos = list(
        map(
            lambda elem:
            ImageInfo(
                path=f"{elem[1][0]}.{ext}",
                json=elem[0],
                label=f"{elem[1][0]}.{label_ext}",
                indir=indir
            ),
            filter(
                lambda elem: elem[1][-1] == json_ext,
                map(
                    lambda elem: (elem, elem.split("__")),
                    file_names
                )
            )
        )
    )
    return image_infos


def gen_labels(image_infos):
    for image_info in image_infos:
        image_segments = derive_image_segments(image_info.json, json_class_to_class_id, W, H)

        out, _ = derive_masks(image_segments, lambda x: cmap[x], len(class_names))

        img = Image.fromarray(out)
        img.save(image_info.label)


def create_pairs(image_infos):
    pairs = [[i.path, i.label] for i in image_infos]
    return pairs


json_class_to_label_map = {
    "line": "yl",
    "w_line": "wl"
}

label_to_index_map = {
    "bg": 0,
    "wl": 1,
    "yl": 2
}

H = 692
W = 1283


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='lighting_variations_data',
                        help='Path to save tfrecord')
    parser.add_argument('--indir', type=str, default='tfrecords',
                        help='Dataset path.')
    parser.add_argument('--target', type=str, default='train',
                        help='train, val, test')
    parser.add_argument('--gen',
                        action='store_false',
                        help='generates labeled images', )
    args = parser.parse_args()

    image_infos = deduce_image_infos(args.indir)

    if args.gen:
        gen_labels(image_infos)

    pairs = create_pairs(image_infos)

    fname = 'camvid-{}.tfrecord'.format(args.target)
    convert_to_tfrecord(pairs, args.outdir, fname)


if __name__ == '__main__':
    main()
