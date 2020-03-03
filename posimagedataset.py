import torch
from  torch.utils.data.dataset import IterableDataset
import numpy as np
import cv2
import os
import scipy.io
from PIL import Image


def standardize_label(label, orim):
    '''

    :param label: the label to standardize
    :param orim: the original image
    :return: standardized label
    '''
    label_ = np.zeros(label.shape)
    label_[0] = label[0] / orim.size[0]
    label_[1] = label[1] / orim.size[1]
    return label_

def destandardize_label(label, orim):
    '''

    :param label: the label to standardize
    :param orim: the original image
    :return: standardized label
    '''
    label_ = label.clone()
    label_[0] = label[0] * orim.size[0]
    label_[1] = label[1] * orim.size[1]
    return label_


class PoseImageDataset(IterableDataset):

    def __init__(self,transforms, imagesize=224, imagespath='', labelsfilepath=''):
        self.imagesize = imagesize
        self.imagespath = imagespath
        self.labelsfilepath = labelsfilepath
        self.filenames = None
        # listing files
        files = None
        for dir, dirnames, filenames in os.walk(self.imagespath):
            self.filenames = filenames
        self.filenames.sort()
        #resetting
        self.reset()
        #loading annotations file into matrix
        self.annotationmat = scipy.io.loadmat(self.labelsfilepath)
        #adapting dimensionality
        joints = self.annotationmat['joints']
        joints = np.swapaxes(joints, 2, 0)
        #free-joints = np.swapaxes(joints, 2, 1)
        self.labels = joints
        self.transforms = transforms

    def reset(self):
        self.counter = 0;

    def __iter__(self):
        #gather image and label
        return self



    def __next__(self):
        if self.counter==10000:
            self.counter = 0
        fn = self.filenames[self.counter]
        orim = Image.open(os.path.join(self.imagespath,fn))
        im = self.transforms(orim)
        label = self.labels[self.counter]
        #standardizing
        label = standardize_label(label, orim)
        label = torch.from_numpy(label)
        label = label.type(torch.FloatTensor)
        self.counter += 1
        return (im, label, fn)

    def len(self):
        return len(self.filenames)



