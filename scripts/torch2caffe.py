import os

import torch
import caffe
import time
import numpy as np

CONV = 0
BN = 1
SCALE = 2
RELU = 3
FC = 4

type_map = {
    'CONV': CONV,
    'BN': BN,
    'SCALE': SCALE,
    'RELU': RELU,
    'FC': FC
}


def save_bn2caffe(running_mean, running_var, bn_param):
    bn_param[0].data[...] = running_mean.numpy()
    bn_param[1].data[...] = running_var.numpy()
    bn_param[2].data[...] = np.array([1.0])


def save_scale2caffe(weights, biases, scale_param):
    scale_param[1].data[...] = biases.numpy()
    scale_param[0].data[...] = weights.numpy()


def save_conv2caffe(weights, biases, conv_param):
    conv_param[0].data[...] = weights.numpy()
    if biases is not None:
        conv_param[1].data[...] = biases.numpy()


def save_fc2caffe(weights, biases, fc_param):
    fc_param[0].data[...] = weights.numpy()
    if biases is not None:
        fc_param[1].data[...] = biases.numpy()


def convert(torch_param_name,
            caffe_param_name,
            torch_weights,
            caffe_weights,
            layer_type):
    params = caffe_weights[caffe_param_name]
    print(f'Converting {torch_param_name} -> {caffe_param_name}')
    if layer_type == CONV:
        save_conv2caffe(torch_weights[f'{torch_param_name}.weight'],
                        torch_weights.get(f'{torch_param_name}.bias', None),
                        params)
    elif layer_type == BN:
        save_bn2caffe(torch_weights[f'{torch_param_name}.running_mean'],
                      torch_weights[f'{torch_param_name}.running_var'],
                      params)
    elif layer_type == SCALE:
        save_scale2caffe(torch_weights[f'{torch_param_name}.weight'],
                         torch_weights[f'{torch_param_name}.bias'],
                         params)
    elif layer_type == FC:
        save_fc2caffe(torch_weights[f'{torch_param_name}.weight'],
                      torch_weights[f'{torch_param_name}.bias'],
                      params)
    else:
        raise ValueError(f'Wrong type: {layer_type}')


if __name__ == '__main__':
    # prototxt = '/Users/troy/Projects/Clion/caffe-lightning/models/light/tiny.ng.deploy.prototxt'
    prototxt = '/Users/troy/Projects/Clion/caffe-lightning/models/light/squeezenet.v11.ss.prototxt'
    save_path = os.path.splitext(prototxt)[0] + '.caffemodel'
    net = caffe.Net(prototxt, caffe.TEST)
    time.sleep(1)
    # weights = 'weights/best_relu.pt'
    # weights = 'weights/best.pt'
    weights = '/Users/troy/Projects/Works/YOLOv3-Ultralytics/weights/r-640-origin-sq-backbone.ss.pt'
    weights = torch.load(weights, map_location='cpu')['model']
    # weights = torch.load(weights, map_location='cpu')
    cfg = 'scripts/squeeze.ss'
    with open(cfg) as fp:
        weights_map = [line.strip().split(' ') for line in fp.readlines()]
        weights_map = [l for l in weights_map if len(l) == 3]

    for tk, ck, t in weights_map:
        convert(tk, ck, weights, net.params, type_map[t])

    net.save(save_path)
