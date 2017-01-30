# -*- coding: utf-8 -*-

import os
import argparse
import random
import pickle
import numpy as np

import chainer
from chainer import training
from chainer import serializers
from chainer.training import extensions

from autoencoder import StackedCAE

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath("__file__")), "../dataset")


class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, random=True):
        self.base = chainer.datasets.ImageDataset(path, root)
        self.mean = mean.astype('f')
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):

        crop_size = self.crop_size
        image, _ = self.base[i]
        _, h, w = image.shape

        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        image -= self.mean[:, top:bottom, left:right]
        image *= (1.0 / 255.0)  # Scale to [0, 1]

        return image, image

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Learning stacked convolutional auto-encoder.')
    parser.add_argument('train', help='Path to training image-label list file')
    parser.add_argument('--batchsize', '-B', type=int, default=32, help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--loaderjob', '-j', type=int, help='Number of parallel data loading processes')
    parser.add_argument('--out', '-o', default='result', help='Out directory')
    parser.add_argument('--root', '-R', default='.', help='Root directory path of image files')
    args = parser.parse_args()

    model = StackedCAE()
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).user()  # Make the GPU current
        model.to_gpu()

    print("[ PREPROCESS ] Load image-path list file.")
    with open(args.train, "rb") as rf:
        unlabeled_image_dataset_list = pickle.load(rf)

    train = PreprocessedDataset()
