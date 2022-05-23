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
    generate_first_line, generate_line, is_inverting_line

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
    def __init__(self, json):
        self.json = json


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
    parser.add_argument('--indir', type=str, default='data/CamVid',
                        help='Dataset path.')
    parser.add_argument('--target', type=str, default='train',
                        help='train, val, test')
    parser.add_argument('--gen',
                        help='generate labels')
    args = parser.parse_args()

    txt_path = os.path.join(args.indir, '{}.txt'.format(args.target))
    pairs = load_path(txt_path, args.indir)

    fname = 'camvid-{}.tfrecord'.format(args.target)
    convert_to_tfrecord(pairs, args.outdir, fname)


if __name__ == '__main__':
    main()
