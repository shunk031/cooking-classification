# -*- coding: utf-8 -*-

import os
import argparse
import pickle
import datetime
import csv
import numpy as np

import chainer

from model import archs
from predict import PreprocessedUnLabeledDataset
from predict import predict
from predict import RESULT_DIR
from logger import create_log_dict, save_log

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Predicting convnet using ensemble network.')
    parser.add_argument('test', help='Path to testing image-label list file')
    parser.add_argument('trained_model0', help='Path to first trained model')
    parser.add_argument('trained_model1', help='Path to seconed trained model')
    parser.add_argument('--arch0', choices=archs.keys(), default='alex')
    parser.add_argument('--arch1', choices=archs.keys(), default='alex')
    parser.add_argument('--root', '-R', default='.')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--mean', '-m', default='train_mean.npy', help='Mean file (computed by compute_mean.py)')
    args = parser.parse_args()

    # logging setting
    now = datetime.datetime.today()
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
    with open(args.test, "rb") as rf:
        test_image_dataset_list = pickle.load(rf)

    if isinstance(test_image_dataset_list[0], tuple):
        test_image_dataset_list = [file[0] for file in test_image_dataset_list]

    print("[ PREPROCESS ] Load the datasets and mean file.")
    mean = np.load(args.mean)
    test_datasets = PreprocessedUnLabeledDataset(test_image_dataset_list, args.root, mean, 227)

    model0.train = model1.train = False
    pred_results = []
    for i in range(len(test_datasets)):
        test_data = test_datasets[i]

        pred0 = predict(model0, test_data, args.gpu)
        pred1 = predict(model1, test_data, args.gpu)
        pred = (pred0 + pred1) / 2
        pred_idx = np.argsort(pred)[0][::-1][0]

        print("Test No. {:5} Predict -> {}".format(i, pred_idx))
        pred_results.append([i, pred_idx])

    if not os.path.isdir(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    with open(os.path.join(RESULT_DIR, now_str + "_result.csv"), "w") as wf:
        writer = csv.writer(wf)

        for result in pred_results:
            writer.writerow(result)

    print("[ FINISH ] Finish predicting at {}".format(now_str))
