# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from preprocess_dataset import PREPROCESS_TYPES
from preprocess_dataset import DATASET_DIR
from preprocess_dataset import CROPPED_ROOT_DIR
from preprocess_dataset import ROTATE_TYPE_DICT

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='dump labeled image dataset list')
    parser.add_argument('preprocess_type', choices=PREPROCESS_TYPES)
    parser.add_argument('--rotate', action="store_true")
    parser.set_defaults(rotate=False)
    args = parser.parse_args()

    # load train file contain image filename and category id
    if args.preprocess_type == "train" or args.preprocess_type == "cae":
        train_df = pd.read_csv(os.path.join(DATASET_DIR, "clf_train_master.tsv"), delimiter="\t")

    else:
        test_df = pd.read_csv(os.path.join(DATASET_DIR, "clf_test.tsv"), delimiter='\t')

    image_dataset_list = []
    if args.preprocess_type == "train":

        train_data, val_data = train_test_split(train_df, test_size=0.1, random_state=0)

        for key, column in train_data.iterrows():
            filename, category_id = column.values
            root, ext = os.path.splitext(filename)
            image_dataset_list.append((os.path.join("train_model", "train", str(category_id), root, "cropped_{}".format(filename)), category_id))
        dump_pickle_name = "train_labeled_image_dataset_list.pkl"

        if args.rotate:
            for key, colum in train_data.iterrows():
                filename, category_id = column.values
                root, ext = os.path.splitext(filename)
                for rotate_type in ROTATE_TYPE_DICT.keys():
                    image_dataset_list.append((os.path.join("train_model", "train", str(category_id), root, "cropped_{}_{}".format(rotate_type, filename)), category_id))
            dump_pickle_name = "train_labeled_image_augment_dataset_list.pkl"

        with open(dump_pickle_name, "wb") as wf:
            pickle.dump(image_dataset_list, wf)

        image_dataset_list = []
        for key, column in val_data.iterrows():
            filename, category_id = column.values
            root, ext = os.path.splitext(filename)
            image_dataset_list.append((os.path.join("train_model", "val", str(category_id), root, "cropped_{}".format(filename)), category_id))
        dump_pickle_name = "val_labeled_image_dataset_list.pkl"

    elif args.preprocess_type == "cae":
        CROPPED_IMAGE_DIR = os.path.join(CROPPED_ROOT_DIR, 'train_cae')

        # get all unlabeled image filenames
        unlabeled_image_dataset_list = os.listdir(CROPPED_IMAGE_DIR)
        image_dataset_list = [os.path.join("train_cae", filename) for filename in unlabeled_image_dataset_list]
        dump_pickle_name = "cae_image_dataset_list.pkl"

    elif args.preprocess_type == "test":

        CROPPED_IMAGE_DIR = os.path.join(CROPPED_ROOT_DIR, "test_model")
        image_dataset_list = [os.path.join("test_model", filename) for filename in os.listdir(CROPPED_IMAGE_DIR)]

        dump_pickle_name = "test_image_dataset_list.pkl"

    # dump image dataset list as pikle file
    with open(dump_pickle_name, "wb") as wf:
        pickle.dump(image_dataset_list, wf)

    print("Dump image dataset list.")
