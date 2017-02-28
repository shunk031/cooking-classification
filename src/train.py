# -*- coding: utf-8 -*-

import os
import argparse
import random
import pickle
import datetime

import numpy as np
from sklearn.model_selection import train_test_split

import chainer
from chainer import training
from chainer import serializers
from chainer.training import extensions

import model
from model import archs
from logger import create_log_dict, save_log

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath("__file__")), "../dataset")


class PreprocessedLabeledDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, random=True):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.mean = mean.astype('f')
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        # It applies following preprocesses:
        #     - Cropping (random or center rectangular)
        #     - Random flip
        #     - Scaling to [0, 1] value

        # print('{}, {}'.format(self.base._root, self.base._pairs[i]))

        crop_size = self.crop_size

        image, label = self.base[i]
        _, h, w = image.shape

        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        image -= self.mean[:, top:bottom, left:right]
        image *= (1.0 / 255.0)  # Scale to [0, 1]
        return image, label


class TestModeEvaluator(extensions.Evaluator):

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('train', help='Path to training image-label list file')
    parser.add_argument('val', help='Path to validation image-label list file')
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='alex',
                        help='Convnet architecture')
    parser.add_argument('--batchsize', '-B', type=int, default=32,
                        help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU')
    parser.add_argument('--initmodel',
                        help='Initialize the model from given file')
    parser.add_argument('--loaderjob', '-j', type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--mean', '-m', default='train_mean.npy',
                        help='Mean file (computed by compute_mean.py)')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--root', '-R', default='.',
                        help='Root directory path of image files')
    parser.add_argument('--val_batchsize', '-b', type=int, default=64,
                        help='Validation minibatch size')
    parser.add_argument('--comment', type=str)
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()

    # logging setting
    now = datetime.datetime.today()
    log_filename_name = "cooking_classification_" + now.strftime('%Y-%m-%d-%H-%M-%S')
    log_filename_ext = ".json"
    log_filename = log_filename_name + log_filename_ext
    save_log(create_log_dict(args), log_filename)

    model = archs[args.arch]()
    if args.initmodel:
        print('[ PREPROCESS ] Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make the GPU current
        model.to_gpu()

    print("[ PREPROCESS ] Load image-label list file.")
    with open(args.train, "rb") as rf:
        train_tuples = pickle.load(rf)
    with open(args.val, "rb") as rf:
        val_tuples = pickle.load(rf)

    num_of_image_data = len(train_tuples) + len(val_tuples)
    print("{:15}All: {}, Train: {}, Val: {}".format("", num_of_image_data, len(train_tuples), len(val_tuples)))

    # Load the datasets and mean file
    print("[ PREPROCESS ] Load the datasets and mean file.")
    mean = np.load(args.mean)
    train = PreprocessedLabeledDataset(train_tuples, args.root, mean, 227)
    val = PreprocessedLabeledDataset(val_tuples, args.root, mean, 227, False)
    # These iterators load the images with subprocesses running in parallel to
    # the training/validation.
    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize, n_processes=args.loaderjob)
    val_iter = chainer.iterators.MultiprocessIterator(
        val, args.val_batchsize, repeat=False, n_processes=args.loaderjob)

    # Set up an optimizer
    print("[ PREPROCESS ] Set up an optimizer.")
    if args.arch == "impresnet":
        optimizer = chainer.optimizers.Adam()
    else:
        optimizer = chainer.optimizers.MomentumSGD(lr=0.001, momentum=0.9)
    optimizer.setup(model)

    # Set up a trainer
    print("[ PREPROCESS ] Set up a trainer.")
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    val_interval = (1000 if args.test else 10000), 'iteration'
    log_interval = (1000 if args.test else 1000), 'iteration'

    trainer.extend(TestModeEvaluator(val_iter, model, device=args.gpu),
                   trigger=val_interval)
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        model, now.strftime('%Y-%m-%d-%H-%M-%S') + "_" + args.arch + "_" + '_model_iter_{.updater.iteration}'), trigger=val_interval)

    # Be careful to pass the interval directly to LogReport
    # (it determines when to emit log rather than when to read observations)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='{}_loss.png'.format(args.arch), trigger=log_interval))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='{}_accuracy.png'.format(args.arch), trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy', 'lr', 'elapsed_time'
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

    # Save the trained model
    serializers.save_npz(os.path.join(args.out, now.strftime('%Y-%m-%d-%H-%M-%S') + "_{}_model_final.npz".format(args.arch)), model)
    serializers.save_npz(os.path.join(args.out, now.strftime('%Y-%m-%d-%H-%M-%S') + "_{}_optimaizer_final.npz".format(args.arch)), optimizer)

    print("[ FINISH ] Training is Finished.")
