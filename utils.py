import torch
# import Image
import numpy as np 
import torch.utils.data as data

from os import listdir
from os.path import join
from scipy.misc import imread, imresize, imsave

import random
from time import time

### image w x h
image_sz = 256

### dict w x h
dict_sz = 286

### Load data set
# read data from data set files
class dataFromFolder(data.Dataset):
    # init
    def __init__(self, data_dir, opt):
        super(dataFromFolder, self).__init__()
        self.photo_path = join(data_dir, "Photos")
        self.sketches_path = join(data_dir, "Sketches")
        self.image_file_names = [x for x in listdir(self.photo_path) if isImageFile(x)]
        self.opt = opt

    # load image pair (photo - sketch)
    def __getitem__(self, index):
        w = random.randint(0, max(0, dict_sz - image_sz - 1))
        h = random.randint(0, max(0, dict_sz - image_sz - 1))
        t = random.random()
        photo = loadImage(join(self.photo_path, self.image_file_names[index]), w, h, t, self.opt)
        sketch = loadImage(join(self.sketches_path, self.image_file_names[index]), w, h, t, self.opt)

        return photo, sketch, index, w, h

    # return data set num
    def __len__(self):
        return len(self.image_file_names)

### get train data
def getTrainData(root_dir, opt = 'Training'):
    train_dir = join(root_dir, "Training")
    return dataFromFolder(train_dir, opt)

### get test data
def getTestData(root_dir, opt = 'Testing'):
    test_dir = join(root_dir, "Testing")
    return dataFromFolder(test_dir, opt)

### Image operatoration
### photo
# load image from data set
def loadImage(file_path, w, h, t, opt):
    img = imread(file_path)
    if len(img.shape) < 3:
        img = np.expand_dims(img, axis = 2)
        img = np.repeat(img, 3, axis = 2)
    
    if opt == 'Training':
        img = imresize(img, (dict_sz, dict_sz))
        img = np.transpose(img, (2, 0, 1))
        # numpy.ndarray to FloatTensor
        img = torch.from_numpy(img)
        img = img[:, h : h + image_sz,
                     w : w + image_sz]

        if t < 0.5:# flip
            idx = [i for i in range(img.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            img = img.index_select(2, idx)

    else: #'Testing'
        img = imresize(img, (image_sz, image_sz))
        img = np.transpose(img, (2, 0, 1))
        # numpy.ndarray to FloatTensor
        img = torch.from_numpy(img)

    img = preProcessImg(img)

    return img

### save image
def saveImage(img, file_name, log=False, size=(250,200,3)):
    img = deProcessImg(img)
    img = img.numpy()
    img *= 255.0
    img = img.clip(0, 255)
    img = np.transpose(img, (1, 2, 0))
    img = imresize(img, size)
    img = img.astype(np.uint8)
    imsave(file_name, img)
    if log:
        print ("Image saved as {}".format(file_name))

### pre process of image
def preProcessImg(img, use_cuda=False):
    # [0,255] image to [0,1]
    if use_cuda:
        img = torch.cuda.FloatTensor(img.size()).copy_(img)
    else:
        img = torch.FloatTensor(img.size()).copy_(img)
    min = 1.0 * img.min()
    max = 1.0 * img.max()
    img.add_(-min).mul_(1.0 / (max - min))

    # RGB to BGR
    # print('pre',type(img),img.shape)
    if use_cuda:
        idx = torch.cuda.LongTensor([2, 1, 0])
    else:
        idx = torch.LongTensor([2, 1, 0])
    img = torch.index_select(img, 0, idx)
    # print('pre',type(img),img.shape)
    # [0,1] to [-1,1]
    img = img.mul_(2).add_(-1)

    # check that input is in expected range
    assert img.max() <= 1, "error img max <= 1"
    assert img.min() >= -1, "error img min >= -1"
    return img

### de process of img
def deProcessImg(img, use_cuda=False):
    # BGR to RGB
    # print('post',type(img),img.shape)
    if use_cuda:
        idx = torch.cuda.LongTensor([2, 1, 0])
    else:
        idx = torch.LongTensor([2, 1, 0])
    img = torch.index_select(img, 0, idx)
    # print('post',type(img),img.shape)
    # [-1,1] to [0,1]
    img = img.add_(1).div_(2)
    return img

### is the image loaded right
def isImageFile(file_name):
    return any(file_name.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

