# -*- coding: utf-8 -*-

import argparse
import os
import sys
import numpy as np
import chainer
from PIL import Image
import pickle


def compute_mean(dataset):

    sum_image = 0
    N = len(dataset)

    for i, (image, _) in enumerate(dataset):
        try:
            sum_image += image
            sys.stderr.write("{}/{}\r".format(i + 1, N))
            sys.stderr.flush()
        except ValueError as err:
            print(err)

    sys.stderr.write("\n")
    return sum_image / N

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compute images mean array')
    parser.add_argument('dataset', help='Path to training image-label list file')
    parser.add_argument('--root', '-R', default='.', help='Root directory path of image files')
    parser.add_argument('--output_path', '-o', default='.', help='path to output mean array')
    parser.add_argument('--cae', action='store_true')
    parser.add_argument('--image', '-i', action='store_true')
    parser.set_defaults(cae=False)
    parser.set_defaults(image=False)
    args = parser.parse_args()

    print("[ LOAD ] Load image-label list file.")
    with open(args.dataset, "rb") as rf:
        image_dataset_list = pickle.load(rf)

    print("[ LOAD ] Load dataset images into LabeledImageDataset.")
    dataset = chainer.datasets.LabeledImageDataset(image_dataset_list, args.root)
    print("[ CALC ] Compute mean image.")
    mean = compute_mean(dataset)

    if args.cae:
        mean_filename = "cae_mean"
    else:
        mean_filename = "train_mean"

    output_path = os.path.join(args.output_path, mean_filename)

    if args.image:
        mean_image = mean.transpose(1, 2, 0).astype(np.uint8)
        mean_image = Image.fromarray(mean_image)
        mean_image.save(mean_filename + "_image.jpg", "JPEG")

    np.save(output_path + ".npy", mean)
