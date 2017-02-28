# -*- coding: utf-8 -*-

import os
import argparse
import pickle
import random
import csv
import datetime
import numpy as np

import chainer

from PIL import Image
from multiprocessing import Pool
from model import archs
from logger import create_log_dict, save_log

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath("__file__")), "../dataset")
RESULT_DIR = os.path.join(os.path.dirname(os.path.realpath("__file__")), "pred_result")


class PreprocessedUnLabeledDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, random=False):
        self.base = chainer.datasets.ImageDataset(path, root)
        self.mean = mean.astype('f')
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):

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


def predict(model, test_data, use_gpu):
    if use_gpu >= 0:
        img = chainer.cuda.cupy.asarray([test_data], dtype=np.float32)
    else:
        img = np.array([test_data])

    pred = model.predict(img).data

    if use_gpu >= 0:
        pred = chainer.cuda.to_cpu(pred)
    else:
        pred = pred.data

    return pred

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Predicting convnet from ILSVRC2012 dataset")
    parser.add_argument('test', help='Path to testing image-label list file')
    parser.add_argument('trained_model', help='Path to traied model')
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='alex')
    parser.add_argument('--root', '-R', default='.')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--mean', '-m', default='train_mean.npy', help='Mean file (computed by compute_mean.py)')
    args = parser.parse_args()

    # logging setting
    now = datetime.datetime.today()
    strnow = now.strftime('%Y-%m-%d-%H-%M-%S')
    log_filename_name = "result_" + strnow
    log_filename_ext = ".json"
    log_filename = log_filename_name + log_filename_ext
    save_log(create_log_dict(args), log_filename)

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
    test_datasets = PreprocessedUnLabeledDataset(test_image_dataset_list, args.root, mean, model.insize)

    model.train = False
    pred_results = []
    for i in range(len(test_datasets)):
        test_data = test_datasets[i]
        pred = predict(model, test_data, args.gpu)

        pred_idx = np.argsort(pred)[0][::-1][0]
        print("test no. {:5}: predict: {}".format(i, pred_idx))
        pred_results.append([i, pred_idx])

    if not os.path.isdir(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    with open(os.path.join(RESULT_DIR, strnow + "_result.csv"), "w") as wf:
        writer = csv.writer(wf)

        for result in pred_results:
            writer.writerow(result)
