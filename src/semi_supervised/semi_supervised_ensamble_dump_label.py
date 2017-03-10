# -*- coding: utf-8 -*-

import argparse
import os
import sys
import pickle
import datetime
import numpy as np

import chainer

sys.path.append(os.pardir)
from model import archs
from predict import PreprocessedUnLabeledDataset
from predict import predict
from logger import create_log_dict, save_log

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('unlabeled_image', help='Path to unlabeling image-label list file')
    parser.add_argument('trained_model0', help='Path to first traied model')
    parser.add_argument('trained_model1', help='Path to second traied model')
    parser.add_argument('--arch0', choices=archs.keys(), default='alex')
    parser.add_argument('--arch1', choices=archs.keys(), default='alex')
    parser.add_argument('--root', '-R', default='.')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--mean', '-m', default='train_mean.npy', help='Mean file (computed by compute_mean.py)')
    args = parser.parse_args()

    # logging setting
    now = datetime.datetime.now()
    now_str = now.strftime('%Y-%m-%d-%H-%M-%S')
    log_filename_name = "result_" + now_str
    log_filename_ext = ".json"
    log_filename = log_filename_name + log_filename_ext
    save_log(create_log_dict(args), log_filename)
    print("[ INFO ] Start predicting image at: {}".format(now_str))

    model0 = archs[args.arch0]()
    model1 = archs[args.arch1]()
    print('[ PREPROCESS ] Load model from {}.'.format(args.trained_model0))
    print('[ PREPROCESS ] Load model from {}.'.format(args.trained_model1))
    chainer.serializers.load_npz(args.trained_model0, model0)
    chainer.serializers.load_npz(args.trained_model1, model1)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model0.to_gpu()
        model1.to_gpu()
    print("[ PREPROCESS ] Load image-path list file.")
    with open(args.unlabeled_image, "rb") as rf:
        unlabeled_image_dataset_list = pickle.load(rf)

    print("[ PREPROCESS ] Load the datasets and mean file.")
    mean = np.load(args.mean)
    unlabeled_datasets = PreprocessedUnLabeledDataset(
        unlabeled_image_dataset_list, args.root, mean, 227)

    model0.train = model1.train = False
    pred_results = []
    num_of_unlabeled_datasets = len(unlabeled_datasets)
    for idx in range(num_of_unlabeled_datasets):
        unlabeled_data = unlabeled_datasets[idx]

        pred0 = predict(model0, unlabeled_data, args.gpu)
        pred1 = predict(model1, unlabeled_data, args.gpu)
        pred = (pred0 + pred1) / 2

        pred_category = np.argsort(pred)[0][::-1][0]
        pred_score = pred[0][pred_category]

        if pred_score > 0.99:
            print("[{:5}/{:5}] Predict: {:2} (from {})".format(
                idx, num_of_unlabeled_datasets, pred_category, unlabeled_image_dataset_list[idx]))
            pred_results.append((unlabeled_image_dataset_list[idx], pred_category))

    dump_pickle_name = "{}_semi_supervised_ensemble_label_list.pkl".format(now_str)
    with open(dump_pickle_name, "wb") as wf:
        pickle.dump(pred_results)

    print("[ INFO ] Dump image dataset list as pickle file ({})".format(now_str))
