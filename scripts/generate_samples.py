import argparse
import os

import numpy as np
import tqdm
from PIL import Image
from numpy.random import choice as pick

from utils.parse_config import parse_data_cfg
from utils.utils import xywh2xyxy


# noinspection DuplicatedCode
class Generator:
    def __init__(self, data_file, samples):
        """
        初始化
        :param data_file:   包含图片列表的 txt
        :param samples:     每张图片最多有多少了 samples
        """
        with open(data_file) as fp:
            files = [f.strip() for f in fp.readlines()]
            files = [f for f in files if len(f) > 0]
        self.origin_file = data_file
        self.out_file = os.path.splitext(data_file)[0] + '.gen.txt'
        if not os.path.exists(self.out_file):
            with open(self.out_file, 'w') as fp:
                pass
        self.files = files
        self.should_generate_nums = samples - len(files)
        self.objects = []

    def read_objects(self):
        """
        读取所有的目标保存在内存中
        :return: None
        """
        assert self.should_generate_nums > 0
        for f in tqdm.tqdm(self.files, 'Reading imgs'):
            # print(f'File: {f}')
            img: Image.Image = Image.open(f)
            label_path = f.replace('images', 'labels').replace(os.path.splitext(f)[-1], '.txt')
            labels = np.loadtxt(label_path)
            if labels.ndim == 1:
                labels = labels[np.newaxis]
            # print(f'Load {f} [{labels.shape[0]}]')
            w, h = img.size
            labels[..., [1, 3]] *= w
            labels[..., [2, 4]] *= h
            labels[..., 1:] = xywh2xyxy(labels[..., 1:])
            labels = labels.astype(np.int)
            for l, xmin, ymin, xmax, ymax in labels:
                self.objects.append((img.crop((xmin, ymin, xmax, ymax)), int(l)))
        print(f'Read {len(self.objects)} objects')

    def generate(self):
        """
        生成目标，随机抓取几个目标，粘贴在随机选中的一张图片中。
        :return:
        """
        idx = 0
        print(f'Should generate extra {self.should_generate_nums} samples')
        for _ in tqdm.tqdm(range(self.should_generate_nums), 'Generating'):
            f = self.files[pick(len(self.files))]
            img: Image.Image = Image.open(f)
            label_path = f.replace('images', 'labels').replace(os.path.splitext(f)[-1], '.txt')
            ori_labels = np.loadtxt(label_path)
            if ori_labels.ndim == 1:
                ori_labels = ori_labels[np.newaxis]
            nums = pick(list(range(1, opt.max)))
            for nb in range(nums):
                obj, clz = self.objects[pick(len(self.objects))]
                obj = obj.convert(img.mode)
                w, h = img.size
                ow, oh = obj.size
                ox, oy = pick(w - ow // 2), pick(h - oh // 2)
                l = np.array([[clz, ox, oy, ow, oh]], dtype=np.float)
                l[..., [1, 3]] /= w
                l[..., [2, 4]] /= h
                xmin = ox - ow // 2
                ymin = oy - oh // 2
                xmax = xmin + ow
                ymax = ymin + oh
                img.paste(obj, (xmin, ymin, xmax, ymax))
                ori_labels = np.concatenate((ori_labels, l))
            out_path = os.path.split(f)[0] + os.sep + f'{opt.prefix}{idx:08d}.jpg'
            np.savetxt(os.path.split(label_path)[0] + os.sep + f'{opt.prefix}{idx:08d}.txt', ori_labels, fmt='%.8f')
            img.save(out_path)
            # print(f'Saving {out_path} with {ori_labels.shape[0]} [{nums}]')
            with open(self.out_file, 'a+') as fp:
                fp.write(out_path + '\n')
            idx += 1

    def remove(self):
        print('Removing extra samples')
        img_dir = os.path.split(self.files[0])
        all_imgs = [f.path for f in os.scandir(img_dir)]
        all_imgs = [f for f in all_imgs if opt.prefix.lower() in f.lower()]
        for img in tqdm.tqdm(all_imgs, 'Removing imgs'):
            # print(img)
            os.remove(img)

        for img in tqdm.tqdm(all_imgs, 'Removing labels'):
            label = img.replace('images', 'labels').replace(os.path.splitext(img)[-1], '.txt')
            try:
                os.remove(label)
            except FileNotFoundError as e:
                pass

        print('last removing from file')
        # with open(self.origin_file, 'w') as fp:
        #     files = [f + '\n' for f in self.files if opt.prefix.lower() in os.path.split(f)[-1].lower()]
        #     fp.writelines(files)
        os.remove(self.out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='data/light_debug.data')
    parser.add_argument('--samples', '-s', type=int, default=260)
    parser.add_argument('--prefix', '-p', type=str, default='generated.')
    parser.add_argument('--max', '-m', type=int, default=5)
    parser.add_argument('--remove', '-r', action='store_true', default=False)
    opt = parser.parse_args()
    print(opt)
    data = parse_data_cfg(opt.data)
    generator = Generator(data['train'], opt.samples)
    if opt.remove:
        generator.remove()
    else:
        generator.read_objects()
        generator.generate()
