# -*- coding: utf-8 -*-

import os

from PIL import Image
import pickle

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath("__file__")), "../dataset")
CROPPED_DIR = os.path.join(DATASET_DIR, "cropped_images")

if not os.path.isdir(CROPPED_DIR):
    print("Create {}".format(CROPPED_DIR))
    os.makedirs(CROPPED_DIR)

if __name__ == '__main__':

    dataset_dir1 = os.path.join(DATASET_DIR, "clf_train_images_labeled_1")
    files = os.listdir(dataset_dir1)
    image_paths = [(os.path.join(dataset_dir1, file), file) for file in files]

    dataset_dir2 = os.path.join(DATASET_DIR, "clf_train_images_labeled_2")
    files = os.listdir(dataset_dir2)
    image_paths.extend([(os.path.join(dataset_dir2, file), file) for file in files])

    num_of_images = len(image_paths)

    for i, img_path in enumerate(image_paths):

        img = Image.open(img_path[0], "r")

        print("{:5}/{:5} Now process: {}".format(i, num_of_images, img_path[1]))

        square_size = min(img.size)
        width, height = img.size

        if width > height:
            top = 0
            bottom = square_size
            left = (width - square_size) / 2
            right = left + square_size
            box = (left, top, right, bottom)
        else:
            left = 0
            right = square_size
            top = (height - square_size) / 2
            bottom = top + square_size
            box = (left, top, right, bottom)

        img = img.crop(box)
        img = img.resize((256, 256), Image.ANTIALIAS)

        cropped_image_path = os.path.join(CROPPED_DIR, "cropped_" + img_path[1])
        img.save(cropped_image_path, "JPEG")
