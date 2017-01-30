# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import pickle

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath("__file__")), "../dataset")
CROPPED_DIR = os.path.join(DATASET_DIR, "cropped_images")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='dump labeled image dataset list')

    train_labeled_data = pd.read_csv(os.path.join(DATASET_DIR, "clf_train_master.tsv"), delimiter="\t")
    test_labeled_data = pd.read_csv(os.path.join(DATASET_DIR, "clf_test.tsv"), delimiter="\t")

    train_filenames = train_labeled_data["file_name"]
    train_labels = train_labeled_data["category_id"]

    train_labeled_image_dataset_list = [
        ("cropped_" + filename, label) for filename, label in zip(train_filenames, train_labels)
    ]

    with open("train_labeled_image_dataset_list.pkl", "wb") as wf:
        pickle.dump(train_labeled_image_dataset_list, wf)

    print("Dump labeled image dataset list.")
