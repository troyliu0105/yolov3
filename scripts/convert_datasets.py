import os
import argparse
import glob
import xml.etree.ElementTree as ET
from os.path import splitext, exists
from random import shuffle

import cv2
import numpy as np


def convert_annotations(label_files):
    tree = ET.parse(label_files)
    root = tree.getroot()
    list_with_all_boxes = []
    for boxes in root.iter('object'):
        for box in boxes.findall("bndbox"):
            xmin = int(box.find("xmin").text)
            ymin = int(box.find("ymin").text)
            xmax = int(box.find("xmax").text)
            ymax = int(box.find("ymax").text)
            label = names[boxes.find('name').text.lower()]
            list_with_all_boxes.append([label, xmin, ymin, xmax, ymax])
    return np.array(list_with_all_boxes, dtype=np.float32)


def save_to_folder(out, pairs, idx):
    img_dir = out + os.sep + 'images'
    label_dir = out + os.sep + 'labels'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    for img_f, label_f in pairs:
        img = cv2.imread(img_f)
        h, w, _ = img.shape
        labels = convert_annotations(label_f)
        labels[..., [1, 3]] /= w
        labels[..., [2, 4]] /= h
        out_img_f = img_dir + os.sep + f'{idx:08d}.jpg'
        out_label_f = label_dir + os.sep + f'{idx:08d}.txt'
        img = img[:int(h * (576 * 180))]
        # img = cv2.resize(img, (640, 192))
        cv2.imwrite(out_img_f, img)
        np.savetxt(out_label_f, labels, '%.6f')
        idx += 1
    return idx


def convert_subdirectories(root, out, ratio, src_format, dst_format, normalize):
    if not root.endswith('/'):
        root = root + '/'
    imgs = [f for ext in ('jpg', 'jpeg', 'png') for f in glob.iglob(root + f'**/*.{ext}', recursive=True)]
    imgs_with_labels = [(img_f, splitext(img_f)[0] + '.xml') for img_f in imgs]
    # imgs_with_labels = [item for item in imgs_with_labels if cv2.imread(item[0]) is not None and exists(item[1])]
    shuffle(imgs_with_labels)
    indices = int(len(imgs_with_labels) * ratio)
    idx = 0
    split = int(len(imgs_with_labels) * ratio)
    idx = save_to_folder(out, imgs_with_labels[:split], idx)
    idx = save_to_folder(out, imgs_with_labels[split:], idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/Users/troy/Projects/Datasets/light/raw')
    parser.add_argument('--out', type=str, default='/Users/troy/Projects/Datasets/light/new')
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--names', type=str, default='data/light.ng.names')
    parser.add_argument('--src_format', type=str, default='xyxy', choices=('xywh', 'xyxy'))
    parser.add_argument('--dst_format', type=str, default='xywh', choices=('xywh', 'xyxy'))
    parser.add_argument('--normalize', action='store_true', default=True)
    opt = parser.parse_args()
    with open(opt.names) as fp:
        names = [f.strip() for f in fp.readlines()]
        names = {n: i for i, n in enumerate(names)}
    convert_subdirectories(opt.root, opt.out, opt.ratio, opt.src_format, opt.dst_format, opt.normalize)
