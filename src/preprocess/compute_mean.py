# -*- coding: utf-8 -*-

import argparse
import os
import sys
import pickle

import numpy as np

import chainer

from PIL import Image
from tqdm import tqdm
from preprocess_dataset import PREPROCESS_TYPES


def compute_mean_l(dataset):

    sum_image = 0
    N = len(dataset)

    for image, _ in tqdm(dataset):
        try:
            sum_image += image
        except ValueError:
            pass

    return sum_image / N


def compute_mean_i(dataset):

    sum_image = 0
    N = len(dataset)

    for image in tqdm(dataset):
        try:
            sum_image += image
        except ValueError:
            pass

    return sum_image / N

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compute images mean array')
    parser.add_argument('dataset', help='Path to training image-label list file')
    parser.add_argument('preprocess_type', choices=PREPROCESS_TYPES)
    parser.add_argument('--root', '-R', default='.', help='Root directory path of image files')
    parser.add_argument('--output_path', '-o', default='.', help='path to output mean array')
    parser.add_argument('--image', '-i', action='store_true')
    parser.set_defaults(image=False)
    args = parser.parse_args()

    print("[ LOAD ] Load image-label list file.")
    with open(args.dataset, "rb") as rf:
        image_dataset_list = pickle.load(rf)

    print("[ LOAD ] Load dataset images.")
    if args.preprocess_type == "train":
        dataset = chainer.datasets.LabeledImageDataset(image_dataset_list, args.root)
        mean_filename = "train_mean"

    elif args.preprocess_type == "cae":
        dataset = chainer.datasets.ImageDataset(image_dataset_list, args.root)
        mean_filename = "cae_mean"
    else:
        print("In the test phase, the image-label list is not necessary.")
        sys.exit()

    print("[ CALC ] Compute mean image.")
    if args.preprocess_type == "train":
        mean = compute_mean_l(dataset)
    else:
        mean = compute_mean_i(dataset)

    output_path = os.path.join(args.output_path, mean_filename)

    if args.image:
        mean_image = mean.transpose(1, 2, 0).astype(np.uint8)
        mean_image = Image.fromarray(mean_image)
        mean_image.save(mean_filename + "_image.jpg", "JPEG")

    np.save(output_path + ".npy", mean)
