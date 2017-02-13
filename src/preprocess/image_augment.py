# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd

from PIL import Image
from multiprocessing import Pool

# dataset root dir
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath("__file__")), "../../dataset")
# cropped dataset root dir
CROPPED_DIR = os.path.join(DATASET_DIR, "cropped_images")
# root dir for train model
TRAIN_MODEL_DIR = os.path.join(CROPPED_DIR, "train_model")
# root dir for labeled images
IMAGE_DIR_PATH = os.path.join(DATASET_DIR, "labeled")


def quadrate_image(img):
    """
    :param PIL.Image.Image img:
    :rtype: PIL.Image.Image
    """

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

    quadrate_img = img.crop(box)

    return quadrate_img


def rotate_image(angle, img_path, category_id, root, ext):

    img = Image.open(img_path, "r")

    resize_size = 600
    start_margin = (resize_size - 256) / 2
    end_margin = 256 + start_margin

    ignore_angles = [0, 90, 180, 270]

    if not(angle in ignore_angles):
        rotate_img = img.rotate(angle, expand=True)
        quadrate_img = quadrate_image(rotate_img)
        resize_img = quadrate_img.resize((resize_size, resize_size), Image.ANTIALIAS)
        crop_img = resize_img.crop((start_margin, start_margin, end_margin, end_margin))

    else:
        quadrate_img = quadrate_image(img)
        crop_img = quadrate_img.resize((256, 256), Image.ANTIALIAS)

    return crop_img


def save_argument_images(args_tuple):

    img_path, category_id, root, ext = args_tuple

    print("[ PROCESS ] Now processing: {}".format(root))

    angles = [angle for angle in range(0, 360, 10)]
    ignore_angles = [0, 90, 180, 270]

    for angle in angles:
        rotate_img = rotate_image(angle, img_path, category_id, root, ext)
        rotate_img_filename = "cropped_{}_rotate_{}{}".format(root, angle, ext)
        rotate_img.save(os.path.join(TRAIN_MODEL_DIR, str(category_id), root, rotate_img_filename), "JPEG")

        if not(angle in ignore_angles):
            flip_rotate_img = rotate_img.transpose(Image.FLIP_LEFT_RIGHT)
            flip_img_filename = "cropped_{}_flip_rotate_{}{}".format(root, angle, ext)
            flip_rotate_img.save(os.path.join(TRAIN_MODEL_DIR, str(category_id), root, flip_img_filename), "JPEG")


def make_category_dir(category_id_list):

    for category_id in category_id_list:
        category_dir = os.path.join(TRAIN_MODEL_DIR, str(category_id))
        os.makedirs(category_dir)


def make_augment_dir(pd_df):

    for key, column in pd_df.iterrows():
        row_data = column.values
        root, ext = os.path.splitext(row_data[0])
        category_dir = os.path.join(TRAIN_MODEL_DIR, str(row_data[1]))

        augment_dir = os.path.join(category_dir, root)
        if not os.path.isdir(augment_dir):
            os.makedirs(augment_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Preprocess image: place image data, rotate these data, and so on.")
    parser.add_argument("clf_train_master", type=str, help="Path to clf_train_master.tsv file.")
    parser.add_argument("--loaderjob", type=int, default=4, help="Number of parallel data loading processes.")
    args = parser.parse_args()

    # Load clf_train_master.tsv
    df = pd.read_csv(args.clf_train_master, delimiter="\t")
    category_id_list = list(set(df["category_id"]))

    # make directory if not exists "dataset/cropped_images"
    if not os.path.isdir(CROPPED_DIR):
        os.makedirs(CROPPED_DIR)
        os.makedirs(TRAIN_MODEL_DIR)

        # make category directory
        make_category_dir(category_id_list)

        # make augment directory
        make_augment_dir(df)

    args_list = []
    for i, (key, column) in enumerate(df.iterrows()):
        filename, category_id = column.values
        # filename = row_data[0]
        root, ext = os.path.splitext(filename)

        if i < 5000:
            labeled_image_dir = os.path.join(IMAGE_DIR_PATH, "clf_train_images_labeled_1")
        else:
            labeled_image_dir = os.path.join(IMAGE_DIR_PATH, "clf_train_images_labeled_2")

        args_list.append((os.path.join(labeled_image_dir, filename), category_id, root, ext))

    # save argument images using multiprocessing
    p = Pool(args.loaderjob)
    p.map(save_argument_images, args_list)
