import caffe
import numpy as np
from functools import reduce


def conv(flops, ipt_size, layer):
    weight = layer.blobs[0].data
    bias = None
    if len(layer.blobs) == 2:
        bias = layer.blobs[1].data
    kernel_size = reduce(lambda a, b: a * b, weight.shape[2:])
    has_bias = 0 if bias is None else 1

    return flops


if __name__ == '__main__':
    prototxt = '/Users/troy/Projects/Clion/caffe-lightning/models/light/tiny.ng.deploy.prototxt'
    net = caffe.Net(prototxt, caffe.TEST)
    flops = 0
    input_size = (3, 192, 640)
    for l in net.layers:
        if l.type == 'Convolution':
            flops = conv(flops, input_size, l)
