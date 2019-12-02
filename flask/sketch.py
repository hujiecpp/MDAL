import os
import sys
sys.path.append('../')
from flask import Flask, render_template, Response
from flask import request, json, jsonify
import base64

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from models import G, D, ResnetGenerator, weightsInit
from utils import isImageFile, loadImage, saveImage, deProcessImg, preProcessImg
import functools
import torch.nn as nn
from time import time, strftime
import numpy as np
import cv2
from scipy.misc import imresize, imsave
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # set gpu
G_USE_CUDA = True

class Sketch(object):
    response_img = b''
    use_cuda = G_USE_CUDA
    cudnn.benchmark = True
    need_trans = True

    IMG_SIZE_IN = 256
    IMG_H = 480 # 250
    IMG_W = 640 # 200
    chanels = 3

    pmodel_dir = "../model/photo_G_1_model.pth"
    smodel_dir = "../model/sketch_G_2_model.pth"
    photo_G_1 = None
    sketch_G_2 = None
    # tp = TimePoint()
    def __init__(self):

        photo_G_1_state_dict = torch.load(self.pmodel_dir)
        self.photo_G_1 = ResnetGenerator(self.chanels, self.chanels, 64, norm_layer=functools.partial(nn.BatchNorm2d, affine=True), use_dropout=True)
        self.photo_G_1.load_state_dict(photo_G_1_state_dict)

        sketch_G_2_state_dict = torch.load(self.smodel_dir)
        self.sketch_G_2 = G(self.chanels, self.chanels, 64)
        self.sketch_G_2.load_state_dict(sketch_G_2_state_dict)

        if self.use_cuda:
            self.photo_G_1 = self.photo_G_1.cuda()
            self.sketch_G_2 = self.sketch_G_2.cuda()
        print('Sketch init')

    def preProcess(self, img):
        # print(img.shape, type(img)) # (480, 640, 3) <class 'numpy.ndarray'>
        # if len(img.shape) < 3: # <class 'numpy.ndarray'>
        #     img = np.expand_dims(img, axis = 2)
        #     img = np.repeat(img, 3, axis = 2)
        # imsave('test_pre.jpg',img)
        # self.tp.start()
        img = imresize(img, (self.IMG_SIZE_IN, self.IMG_SIZE_IN))
        img = np.transpose(img, (2, 0, 1))
        # numpy.ndarray to FloatTensor
        img = torch.from_numpy(img)
        # self.tp.getCost('pre_resize')
        if self.use_cuda:
            img = img.cuda()
        # self.tp.getCost('pre_cuda')
        img = preProcessImg(img, self.use_cuda)
        # self.tp.getCost('pre_process') # major cost
        img = Variable(img).view(1, -1, self.IMG_SIZE_IN, self.IMG_SIZE_IN)
        return img

    def postProcess(self, img_gpu):
        # self.tp.start()
        img_gpu = deProcessImg(img_gpu.data[0], self.use_cuda)
        # self.tp.getCost('post_process') # major cost
        img = img_gpu.cpu() # <class 'torch.Tensor'>
        # self.tp.getCost('post_cpu')
        img = img.numpy()
        img *= 255.0
        img = img.clip(0, 255)
        img = np.transpose(img, (1, 2, 0))
        img = imresize(img, (self.IMG_H, self.IMG_W, 3)) # row col
        img = img.astype(np.uint8)
        # self.tp.getCost('post_numpy')
        # imsave('test_post.jpg', img)
        # print(img.shape, type(img)) # (250, 200, 3) <class 'numpy.ndarray'> 
        return img


class TimePoint(object):
    init = None
    pre = None
    cur = None
    def __init__(self):
        super(object,self).__init__()

    def start(self):
        self.init = time()
        self.pre = time()
        self.cur = time()

    def getCost(self, str=''):
        self.cur = time()
        print('cost_{}'.format(str), self.cur-self.pre)
        self.pre = self.cur

    def getTotalCost(self, str=''):
        self.cur = time()
        print('total cost {}'.format(str), self.cur-self.init)