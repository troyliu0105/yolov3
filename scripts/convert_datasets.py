import shutil
import glob
import argparse
import cv2
import numpy as np

import xml.etree.ElementTree as ET
from os.path import splitext, exists
from random import shuffle


def convert_annotations(label_files, label_map):
    tree = ET.parse(label_files)
    root = tree.getroot()
    list_with_all_boxes = []
    for boxes in root.iter('object'):
        ymin, xmin, ymax, xmax = None, None, None, None
        for box in boxes.findall("bndbox"):
            ymin = int(box.find("ymin").text)
            xmin = int(box.find("xmin").text)
            ymax = int(box.find("ymax").text)
            xmax = int(box.find("xmax").text)
        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)
    return np.array(list_with_all_boxes, dtype=np.float32)


def save_to_folder(out, pairs, idx):
    pass


def convert_subdirectories(root, out, ratio, src_format, dst_format, normalize):
    imgs = [f for ext in ('jpg', 'jpeg', 'png') for f in glob.iglob(root + f'**/*.{ext}', recursive=True)]
    imgs_with_labels = [(img_f, splitext(img_f)[0] + '.xml') for img_f in imgs]
    imgs_with_labels = [item for item in imgs_with_labels if cv2.imread(item[0]) is not None and exists(item[1])]
    shuffle(imgs_with_labels)
    indices = int(len(imgs_with_labels) * ratio)
    idx = 0
    # save_to_folder()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--src_format', type=str, default='xyxy', choices=('xywh', 'xyxy'))
    parser.add_argument('--dst_format', type=str, default='xywh', choices=('xywh', 'xyxy'))
    parser.add_argument('--normalize', type=bool, action='store_true', default=True)
    opt = parser.parse_args()
    convert_subdirectories(opt.root, opt.out, opt.ratio, opt.src_format, opt.dst_format, opt.normalize)
