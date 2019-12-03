import os

import torch
import caffe
import time
import numpy as np


def save_bn2caffe(running_mean, running_var, bn_param):
    bn_param[0].data[...] = running_mean.numpy()
    bn_param[1].data[...] = running_var.numpy()
    bn_param[2].data[...] = np.array([1.0])


def save_scale2caffe(weights, biases, scale_param):
    scale_param[1].data[...] = biases.numpy()
    scale_param[0].data[...] = weights.numpy()


def save_conv2caffe(weights, biases, conv_param):
    if biases is not None:
        conv_param[1].data[...] = biases.numpy()
    conv_param[0].data[...] = weights.numpy()


def save_fused_cbn(torch_idx, caffe_name, weights, net):
    caffe_conv_params = net.params[caffe_name]
    conv_key = f'module_list.{torch_idx}.Conv2d.weight'
    conv_key = conv_key if conv_key in weights else f'module_list.{torch_idx}.0.weight'
    bias_key = f'module_list.{torch_idx}.Conv2d.bias'
    bias_key = bias_key if bias_key in weights else f'module_list.{torch_idx}.0.bias'
    conv = weights[conv_key].numpy()
    bias = weights[bias_key].numpy()
    caffe_conv_params[0].data[...] = conv
    caffe_conv_params[1].data[...] = bias


def save_cbn(torch_idx, caffe_name, weights, net):
    conv_key = f'module_list.{torch_idx}.Conv2d.weight'
    bias_key = f'module_list.{torch_idx}.Conv2d.bias'
    if bias_key not in weights:  # 带有 BN 层
        bn_weight = weights[f'module_list.{torch_idx}.BatchNorm2d.weight'].numpy()
        bn_bias = weights[f'module_list.{torch_idx}.BatchNorm2d.bias'].numpy()
        bn_mean = weights[f'module_list.{torch_idx}.BatchNorm2d.running_mean'].numpy()
        bn_var = weights[f'module_list.{torch_idx}.BatchNorm2d.running_var'].numpy()
        conv_weight = weights[conv_key].numpy()
        net.params[caffe_name][0].data[...] = conv_weight
        net.params[f'{caffe_name}/bn'][0].data[...] = bn_mean
        net.params[f'{caffe_name}/bn'][1].data[...] = bn_var
        net.params[f'{caffe_name}/bn'][2].data[...] = np.array([1.0])
        net.params[f'{caffe_name}/scale'][0].data[...] = bn_weight
        net.params[f'{caffe_name}/scale'][1].data[...] = bn_bias
    else:
        conv_weight = weights[conv_key].numpy()
        conv_bias = weights[bias_key].numpy()
        net.params[caffe_name][0].data[...] = conv_weight
        net.params[caffe_name][1].data[...] = conv_bias


if __name__ == '__main__':
    prototxt = '/Users/troy/Projects/Clion/caffe-lightning/models/light/tiny.deploy.fuse.prototxt'
    save_path = os.path.splitext(prototxt)[0] + '.caffemodel'
    # weights = 'weights/best_relu.pt'
    weights = 'weights/best_relu_fused.pt'
    weights_map = [
        # torch_idx, caffe_name
        (0, 'conv0'),
        (2, 'conv1'),
        (4, 'conv2'),
        (6, 'conv3'),
        (8, 'conv4'),
        (9, 'conv5'),
        (10, 'conv6'),
        (11, 'conv7'),
        (12, 'yolo_conv_m'),
        (13, 'yolo_out_m'),
        (16, 'conv8'),
        (19, 'yolo_conv_s'),
        (20, 'yolo_out_s'),
    ]
    weights = torch.load(weights, map_location='cpu')['model']
    net = caffe.Net(prototxt, caffe.TEST)
    time.sleep(1)
    for k in net.params:
        print(k)
    for idx, name in weights_map:
        save_fused_cbn(idx, name, weights, net)
        # save_cbn(idx, name, weights, net)
    net.save(save_path)
