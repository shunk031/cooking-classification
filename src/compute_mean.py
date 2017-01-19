# -*- coding: utf-8 -*-

import argparse
import sys
import numpy as np
import chainer
from PIL import Image
import pickle


def compute_mean(dataset):

    print("compute mean image.")

    sum_image = 0
    N = len(dataset)

    for i, (image, _) in enumerate(dataset):

        sum_image += image
        sys.stderr.write("{}/{}\r".format(i, N))
        sys.stderr.flush()

    sys.stderr.write("\n")
    return sum_image / N

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compute images mean array')
    parser.add_argument('dataset',
                        help='Path to training image-label list file')
    parser.add_argument('--root', '-R', default='.',
                        help='Root directory path of image files')
    parser.add_argument('--output', '-o', default='mean.npy',
                        help='path to output mean array')
    args = parser.parse_args()

    with open(args.dataset, "rb") as rf:
        labeled_image_dataset_list = pickle.load(rf)

    dataset = chainer.datasets.LabeledImageDataset(labeled_image_dataset_list, args.root)
    mean = compute_mean(dataset)
    np.save(args.output, mean)
