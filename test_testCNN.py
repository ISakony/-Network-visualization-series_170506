# -*- coding: utf-8 -*-
import numpy as np
import math
import chainer
from chainer import functions as F
from chainer import links as L
from chainer import \
     cuda, gradient_check, optimizers, serializers, utils, \
     Chain, ChainList, Function, Link, Variable

import os
from StringIO import StringIO
from chainer import function
import numpy
from chainer.utils import type_check
from chainer.functions.loss import mean_squared_error 

from PIL import Image
from chainer import variable
#import cupy as cp
import pickle
import testCNN


import gzip
import os
import struct

import six

from chainer.dataset import download
from chainer.datasets import tuple_dataset


def read_image(path):
    image = np.asarray(Image.open(path)).transpose(2, 0, 1)
    image = image[:, :, :].astype(np.float32)
    image /= 255
    return image



def write_image(image, path):
    print image.max()
    if image.max() >= 1.0:
        image *= image.max()
    image *= 255
    #image = image.transpose(1, 2, 0)
    image = image.astype(np.uint8)
    img = np.zeros((3,image.shape[1],image.shape[2]),dtype = np.uint8)
    print image.shape
    img[1] = image[0]
    #for i in range(3):
    #    img[i] = image[0]
    print img.max()
    img = img.transpose(1, 2, 0)
    result = Image.fromarray(img)
    result.save(path)


out_model_dir = './out_models'

try:
    os.mkdir(out_model_dir)
except:
    pass
      

batchsize=1
n_epoch=1
n_epoch2=1
n_train=1

## MNISTデータをロード
print "load MNIST dataset"
train_data, test_data = chainer.datasets.get_mnist(ndim=3)


model = testCNN.testCNN()
#model.to_gpu()
serializers.load_npz("out_models/train_mm_1_9.npz",model)



def test(epoch,batchsize,train_data,test_data,mod):
    #test task
    x = np.ndarray((batchsize, 1, 28, 28), dtype=np.float32)
    y = np.ndarray((batchsize,), dtype=np.int32)
    for j in range(batchsize):
        rnd = np.random.randint(len(test_data))
        path = test_data[rnd][0] 
        label = test_data[rnd][1]
        x[j] = np.array(path)
        y[j] = np.array(label)

    #x = chainer.Variable(cuda.to_gpu(x))
    #y = chainer.Variable(cuda.to_gpu(y))
    x = chainer.Variable(x)
    y = chainer.Variable(y)

    acc_te = mod.predict(x,y)
    acc_te = mod.accuracy.data
    vis = mod.visual_mask()
    print vis.shape
    write_image(vis[0],"out_images/test.png")
    write_image(x.data[0],"out_images/input.png")

    print 'epoch',epoch,"acc_te",acc_te
    
for epoch in xrange(0,n_epoch):
    for i in xrange(0, n_train, batchsize):    
        test(epoch,batchsize,train_data,test_data,model)
    serializers.save_npz("%s/train_mm_1_%d.npz"%(out_model_dir, epoch),model)



























