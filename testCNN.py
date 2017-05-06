# -*- coding: utf-8 -*-
import numpy as np
import math
import chainer
from chainer import functions as F
from chainer import links as L
from chainer import \
     cuda, gradient_check, optimizers, serializers, utils, \
     Chain, ChainList, Function, Link, Variable
from PIL import Image
import os
from StringIO import StringIO
import math
from chainer import function
import numpy
from chainer.utils import type_check
from chainer.functions.loss import mean_squared_error 
import pylab
from chainer import variable



import math
from chainer.functions.activation import log_softmax


def read_image(path):
    image = np.asarray(Image.open(path)).transpose(2, 0, 1)
    image = image[:, :, :].astype(np.float32)
    image /= 255
    return image

def write_image(image, path):
    image *= 255
    image = image.transpose(1, 2, 0)
    image = image.astype(np.uint8)
    result = Image.fromarray(image)
    result.save(path)


class testCNN(chainer.Chain):

    insize = 28

    def __init__(self):
        w = math.sqrt(2)
        layers = {}
        layers["conv1"] = L.Convolution2D(1,   96, 4, stride=2, pad=1)
        layers["conv2"] = L.Convolution2D(96,  256,  4, stride=2, pad=1)
        layers["conv3"] = L.Convolution2D(256,  384,  3, stride=1, pad=1)
        layers["conv4"] = L.Convolution2D(384, 11,  3, stride=1, pad=1)

        
        super(testCNN, self).__init__(**layers)
        self.train = True
        self.initialW = np.ones((1, 1, 4, 4)).astype(np.float32)#out_c,in_c
        self.averageL0 = np.zeros((1, 1, 28, 28)).astype(np.float32)
        self.averageL1 = np.zeros((1, 1, 14, 14)).astype(np.float32)
        self.averageL2 = np.zeros((1, 1, 7, 7)).astype(np.float32)
        self.averageL3 = np.zeros((1, 1, 7, 7)).astype(np.float32)
        self.averageL4 = np.zeros((1, 1, 7, 7)).astype(np.float32)

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x, t):
        self.clear()
        h = F.leaky_relu(self.conv1(x))
        #print h.data.shape #320,96,14,14
        h = F.leaky_relu(self.conv2(h))
        #print h.data.shape #320,256,7,7
        h = F.leaky_relu(self.conv3(h))
        #print h.data.shape #384,7,7
        h = F.leaky_relu(self.conv4(h))
        #print h.data.shape #11,7,7
        h = F.reshape(F.average_pooling_2d(h, h.data.shape[2]), (x.data.shape[0], 11))
        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)
        self.h = h
        return self.loss

    def predict(self, x, t):#batchsize = 1
        self.clear()
        h = F.leaky_relu(self.conv1(x))
        print len(h.data[0])
        for i in range(len(h.data[0])):
            self.averageL1[0][0] += h.data[0][i]
        self.averageL1 /= len(h.data[0])
        h = F.leaky_relu(self.conv2(h))
        print len(h.data[0])
        for i in range(len(h.data[0])):
            self.averageL2[0][0] += h.data[0][i]
        self.averageL2 /= len(h.data[0])
        h = F.leaky_relu(self.conv3(h))
        print len(h.data[0])
        for i in range(len(h.data[0])):
            self.averageL3[0][0] += h.data[0][i]
        self.averageL3 /= len(h.data[0])
        h = F.leaky_relu(self.conv4(h))
        print len(h.data[0])
        for i in range(len(h.data[0])):
            self.averageL4[0][0] += h.data[0][i]
        self.averageL4 /= len(h.data[0])
        
        h = F.reshape(F.average_pooling_2d(h, h.data.shape[2]), (x.data.shape[0], 11))
        self.accuracy = F.accuracy(h, t)
        
        return h

    def visual_mask(self):
        z = self.averageL4 * self.averageL3
        z = z * self.averageL2
        z = F.deconvolution_2d(Variable(z),self.initialW,stride = 2,pad=1).data * self.averageL1
        z = F.deconvolution_2d(Variable(z),self.initialW,stride = 2,pad=1).data
        return z
    
        
        
        
        
        
        
        
        
        
        
        