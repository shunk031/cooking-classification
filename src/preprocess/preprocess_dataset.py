# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import pandas as pd

from PIL import Image
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
from image_augment import quadrate_image

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath("__file__")), "../../dataset")
CROPPED_ROOT_DIR = os.path.join(DATASET_DIR, "cropped_images")

PREPROCESS_TYPES = ["train", "cae", "test"]


def make_category_dir(category_id_list, root_dir):

    for category_id in category_id_list:
        category_dir = os.path.join(root_dir, str(category_id))
        if not os.path.isdir(category_dir):
            os.makedirs(category_dir)


def make_augment_dir(pd_df, root_dir):

    for key, column in pd_df.iterrows():
        row_data = column.values
        root, ext = os.path.splitext(row_data[0])
        category_dir = os.path.join(root_dir, str(row_data[1]))

        augment_dir = os.path.join(category_dir, root)
        if not os.path.isdir(augment_dir):
            os.makedirs(augment_dir)


def crop_and_save(image_info_tuple):

    save_dir, image_full_path, image_name = image_info_tuple
    print("[ PREPROCESS ] Now processing: {}".format(image_name))
    try:
        # load image as PIL object
        img = Image.open(image_full_path, "r")
        img_array = np.array(img)

        # raise exception if image contains alpha channel
        if img_array.shape[-1] > 3:
            raise ValueError("This image contains alpha channel: {}".format(image_full_path))

        # quadrate and crop image
        crop_img = quadrate_image(img).resize((256, 256), Image.ANTIALIAS)
        cropped_image_path = os.path.join(save_dir, "cropped_{}".format(image_name))
        # save crop image
        crop_img.save(cropped_image_path, "JPEG")

    except ValueError as err:
        print("[ EXCEPTION ] Exception occured: {}".format(err))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("preprocess_type", choices=PREPROCESS_TYPES)
    parser.add_argument("--loaderjob", type=int, default=1)
    args = parser.parse_args()

    # use Pool of multiprocessing module
    p = Pool(args.loaderjob)

    if args.preprocess_type == "train":
        # set cropped image dir
        CROPPED_IMAGE_DIR = os.path.join(CROPPED_ROOT_DIR, "train_model")
        CROPPED_TRAIN_IMAGE_DIR = os.path.join(CROPPED_IMAGE_DIR, "train")
        CROPPED_VAL_IMAGE_DIR = os.path.join(CROPPED_IMAGE_DIR, "val")

        # load clf_tran_master.tsv
        train_df = pd.read_csv(os.path.join(DATASET_DIR, "clf_train_master.tsv"), delimiter="\t")
        # get category id
        category_id_list = list(set(train_df["category_id"]))

        # split to train and validation data
        train_data, val_data = train_test_split(train_df, test_size=0.1, random_state=0)

        # make category and augment directory
        make_category_dir(category_id_list, CROPPED_TRAIN_IMAGE_DIR)
        make_category_dir(category_id_list, CROPPED_VAL_IMAGE_DIR)

        make_augment_dir(train_data, CROPPED_TRAIN_IMAGE_DIR)
        make_augment_dir(val_data, CROPPED_VAL_IMAGE_DIR)

    elif args.preprocess_type == "cae":
        # set cropped image dir
        CROPPED_IMAGE_DIR = os.path.join(CROPPED_ROOT_DIR, "train_cae")
        image_dir_path = os.path.join(DATASET_DIR, "unlabeled")

    elif args.preprocess_type == "test":
        # set cropped_image dir
        CROPPED_IMAGE_DIR = os.path.join(CROPPED_ROOT_DIR, "test_model")
        image_dir_path = os.path.join(DATASET_DIR, "test")

    # augment train data and prepare validation data
    if args.preprocess_type == "train":

        image_dir_path = os.path.join(DATASET_DIR, "labeled")

        # preprocess for train data

        # preprocess for validation data
        image_info_tuple_list = []
        for key, column in val_data.iterrows():
            filename, category_id = column.values
            root, ext = os.path.splitext(filename)

            augment_dir = os.path.join(CROPPED_VAL_IMAGE_DIR, str(category_id), root)

            if os.path.isfile(os.path.join(image_dir_path, "clf_train_images_labeled_1", filename)):
                image_info_tuple_list.append((augment_dir, os.path.join(image_dir_path, "clf_train_images_labeled_1", filename), filename))
            else:
                image_info_tuple_list.append((augment_dir, os.path.join(image_dir_path, "clf_train_images_labeled_2", filename), filename))

        # crop images in multi process
        p.map(crop_and_save, image_info_tuple_list)

    # prepare cae data(or test data)
    else:
        # make train_cae(or test_model) directory if not exists
        if not os.path.isdir(CROPPED_IMAGE_DIR):
            print("Create {}".format(CROPPED_IMAGE_DIR))
            os.makedirs(CROPPED_IMAGE_DIR)

        image_dirs = os.listdir(image_dir_path)
        image_info_tuple_list = []
        for image_dir in image_dirs:
            dataset_dir = os.path.join(image_dir_path, image_dir)
            if os.path.isdir(dataset_dir):
                images = os.listdir(dataset_dir)
                for image in images:
                    image_info_tuple_list.append((CROPPED_IMAGE_DIR, os.path.join(dataset_dir, image), image))

        # crop images in multi process
        p.map(crop_and_save, image_info_tuple_list)
