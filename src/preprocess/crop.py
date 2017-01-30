# -*- coding: utf-8 -*-

import os
import sys
import argparse
import pickle

from PIL import Image

# dataset root dir
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath("__file__")), "../../dataset")
# cropped dataset root dir
CROPPED_DIR = os.path.join(DATASET_DIR, "cropped_images")


def crop_image(img_path):
    """
    crop image to 256 x 256 size
    :param str img_path: path to image
    :rtype: PIL.Image.Image
    """

    img = Image.open(img_path, "r")
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

    crop_img = img.crop(box).resize((256, 256), Image.ANTIALIAS)

    return crop_img


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='crop images to 256x256 size')
    parser.add_argument('--cae', action='store_true')
    parser.set_defaults(cae=False)
    args = parser.parse_args()

    # preprocess for CAE training
    if args.cae:
        CROPPED_IMAGE_DIR = os.path.join(CROPPED_DIR, 'train_cae')
        image_dir_path = os.path.join(DATASET_DIR, 'unlabeled')

    # preprocess for classification model training
    else:
        CROPPED_IMAGE_DIR = os.path.join(CROPPED_DIR, 'train_model')
        image_dir_path = os.path.join(DATASET_DIR, "labeled")

    image_dirs = os.listdir(image_dir_path)
    image_path_list = []

    for image_dir in image_dirs:
        dataset_dir = os.path.join(image_dir_path, image_dir)

        # excluding "gitkeep"
        if os.path.isdir(dataset_dir):
            # print(dataset_dir)
            images = os.listdir(dataset_dir)
            for image in images:
                # print(image)
                image_path_list.append((os.path.join(dataset_dir, image), image))

    # make cropped image dir if not exists
    if not os.path.isdir(CROPPED_IMAGE_DIR):
        print("Create {}".format(CROPPED_IMAGE_DIR))
        os.makedirs(CROPPED_IMAGE_DIR)

    num_of_images = len(image_path_list)
    for i, img_path in enumerate(image_path_list):

        # crop image
        crop_img = crop_image(img_path[0])
        # save cropped image
        cropped_image_path = os.path.join(CROPPED_IMAGE_DIR, "cropped_" + img_path[1])
        crop_img.save(cropped_image_path, "JPEG")

        sys.stderr.write("{}/{} Now process: {}\r".format(i + 1, num_of_images, img_path[1]))
        sys.stderr.flush()

    sys.stderr.write("\n")
