# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import pickle

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath("__file__")), "../../dataset")
CROPPED_DIR = os.path.join(DATASET_DIR, "cropped_images")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='dump labeled image dataset list')
    parser.add_argument('--cae', action='store_true')
    parser.set_defaults(cae=False)
    args = parser.parse_args()

    # load train file contain image filename and category id
    train_labeled_data = pd.read_csv(os.path.join(DATASET_DIR, "clf_train_master.tsv"), delimiter="\t")

    train_filenames = train_labeled_data["file_name"]
    train_labels = train_labeled_data["category_id"]

    if args.cae:
        CROPPED_IMAGE_DIR = os.path.join(CROPPED_DIR, 'train_cae')

        # get all unlabeled image filenames
        unlabeled_image_dataset_list = os.listdir(CROPPED_IMAGE_DIR)
        image_dataset_list = [(filename, 0) for filename in unlabeled_image_dataset_list]
        dump_pkl_filename = "cae_image_dataset_list.pkl"

    else:
        # get all tuples of filename and category id
        image_dataset_list = [
            ("cropped_" + filename, label) for filename, label in zip(train_filenames, train_labels)
        ]
        dump_pkl_filename = "labeled_image_dataset_list.pkl"

    # dump image dataset list as pikle file
    with open(dump_pkl_filename, "wb") as wf:
        pickle.dump(image_dataset_list, wf)

    print("Dump image dataset list.")
