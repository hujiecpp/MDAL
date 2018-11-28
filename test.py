import os
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from models import G, D, ResnetGenerator, weightsInit
from utils import isImageFile, loadImage, saveImage

import functools
import torch.nn as nn

from time import time

### Training parament setting
parser = argparse.ArgumentParser(description = '.. implementation')
parser.add_argument('--dataset', required = True, help = 'CUHKStudent')
parser.add_argument('--cuda', action = 'store_true', help = 'use cuda?')
parser.add_argument('--threads', type = int, default = 4, help = 'number of threads for data loader to use')
parser.add_argument('--p_model', type=str, required=True, help='model file to use')
parser.add_argument('--s_model', type=str, required=True, help='model file to use')
# parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--ngf', type = int, default = 64, help = 'generator filters in first conv layer')
opt = parser.parse_args()
print(opt)
### batch size
batch_size = 1
### RGB chanels
chanels = 3
### image size
image_sz = 256

### cuda setting
if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

### uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms
cudnn.benchmark = True

pmodel_dir = "checkpoint/{}/{}".format(opt.dataset, opt.p_model)
photo_G_1_state_dict = torch.load(pmodel_dir)
# photo_G_1 = G(chanels, chanels, opt.ngf)
photo_G_1 = ResnetGenerator(chanels, chanels, opt.ngf, norm_layer=functools.partial(nn.BatchNorm2d, affine=True), use_dropout=True)
photo_G_1.load_state_dict(photo_G_1_state_dict)

smodel_dir = "checkpoint/{}/{}".format(opt.dataset, opt.s_model)
sketch_G_2_state_dict = torch.load(smodel_dir)
# sketch_G_2 = ResnetGenerator(chanels, chanels, opt.ngf, norm_layer=functools.partial(nn.BatchNorm2d, affine=True), use_dropout=True)
sketch_G_2 = G(chanels, chanels, opt.ngf)
sketch_G_2.load_state_dict(sketch_G_2_state_dict)

image_dir = "dataset/{}/Testing/Photos/".format(opt.dataset)
image_filenames = [x for x in os.listdir(image_dir) if isImageFile(x)]

all_time = 0
num = 0
for image_name in image_filenames:
    num = num + 1
    img = loadImage(image_dir + image_name, -1, -1, -1, 'Testing')
    input = Variable(img).view(1, -1, image_sz, image_sz)
    if opt.cuda:
        # netaG = netaG.cuda()
        photo_G_1 = photo_G_1.cuda()
        sketch_G_2 = sketch_G_2.cuda()
        input = input.cuda()

    start = time()
    wp = photo_G_1(input)
    out = sketch_G_2(wp)
    stop = time()
    all_time = all_time + (stop - start)
    # out = netaG(input)
    #print(out.data.shape)
    out = out.cpu()
    out_img = out.data[0]
    #print(out_img.shape)
    if not os.path.exists("result"):
        os.mkdir("result")
    if not os.path.exists(os.path.join("result", opt.dataset)):
        os.mkdir(os.path.join("result", opt.dataset))
    saveImage(out_img, "result/{}/{}".format(opt.dataset, image_name))

print(all_time, all_time / num)
