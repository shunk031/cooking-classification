# -*- coding: utf-8 -*-

import os
import argparse
import pickle
import random
import csv
import numpy as np

from PIL import Image
from multiprocessing import Pool

import chainer

import model

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath("__file__")), "../dataset")

archs = {
    'alex': model.AlexNet,
    'alexlike': model.AlexLikeNet,
    'deepalexlike': model.DeepAlexLikeNet,
    'resnet': model.ResNet,
}


class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, random=False):
        self.base = chainer.datasets.ImageDataset(path, root)
        self.mean = mean.astype('f')
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        # It applies following preprocesses:
        #     - Cropping (random or center rectangular)
        #     - Random flip
        #     - Scaling to [0, 1] value

        # print('{}, {}'.format(self.base._root, self.base._pairs[i]))

        crop_size = self.crop_size

        image = self.base[i]
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

        return image

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Predicting convnet from ILSVRC2012 dataset")
    parser.add_argument('test', help='Path to testing image-label list file')
    parser.add_argument('trained_model', help='Path to traied model')
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='alex')
    parser.add_argument('--root', '-R', default='.')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--mean', '-m', default='train_mean.npy', help='Mean file (computed by compute_mean.py)')
    args = parser.parse_args()

    model = archs[args.arch]()
    print('[ PREPROCESS ] Load model from {}'.format(args.trained_model))
    chainer.serializers.load_npz(args.trained_model, model)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    print("[ PREPROCESS ] Load image-path list file.")
    with open(args.test, "rb") as rf:
        test_image_dataset_list = pickle.load(rf)

    if isinstance(test_image_dataset_list[0], tuple):
        test_image_dataset_list = [file[0] for file in test_image_dataset_list]

    print("[ PREPROCESS ] Load the datasets and mean file.")
    mean = np.load(args.mean)
    # print("mean shape".format(mean.shape))
    test_datasets = PreprocessedDataset(test_image_dataset_list, args.root, mean, model.insize)

    model.train = False
    pred_results = []
    for i in range(len(test_datasets)):
        test_data = test_datasets[i]

        if args.gpu >= 0:
            img = chainer.cuda.cupy.asarray([test_data], dtype=np.float32)
        else:
            img = np.array([test_data])

        pred = model.predict(img).data

        if args.gpu >= 0:
            pred = chainer.cuda.to_cpu(pred)
        else:
            pred = pred.data

        pred_idx = np.argsort(pred)[0][::-1][0]
        print("test no. {:5}: predict: {}".format(i, pred_idx))
        pred_results.append([i, pred_idx])

    with open("result.csv", "w") as wf:
        writer = csv.writer(wf)

        for result in pred_results:
            writer.writerow(result)
