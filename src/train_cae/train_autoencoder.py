# -*- coding: utf-8 -*-

import os
import argparse
import random
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

import chainer
from chainer import training
from chainer import serializers
from chainer.training import extensions

from autoencoder import StackedCAE

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath("__file__")), "../dataset")


class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, random=True):
        self.base = chainer.datasets.ImageDataset(path, root)
        self.mean = mean.astype('f')
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):

        crop_size = self.crop_size
        image = self.base[i]
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

        return image, image


class TestModeEvaluator(extensions.Evaluator):

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Learning stacked convolutional auto-encoder.')
    parser.add_argument('train', help='Path to training image-label list file')
    parser.add_argument('--batchsize', '-B', type=int, default=32, help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--loaderjob', '-j', type=int, help='Number of parallel data loading processes')
    parser.add_argument('--mean', '-m', default='cae_mean.npy', help='Mean file (computed by compute_mean.py)')
    parser.add_argument('--out', '-o', default='result', help='Out directory')
    parser.add_argument('--root', '-R', default='.', help='Root directory path of image files')
    parser.add_argument('--val_batchsize', '-b', type=int, default=250, help='Validation minibatch size')
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()

    model = StackedCAE()
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make the GPU current
        model.to_gpu()

    print("[ PREPROCESS ] Load image-path list file.")
    with open(args.train, "rb") as rf:
        unlabeled_image_dataset_list = pickle.load(rf)
    unlabeled_image_dataset_list = [dataset_tuple[0] for dataset_tuple in unlabeled_image_dataset_list]

    print("[ PREPROCESS ] Split train and test image.")
    train_images, test_images = train_test_split(
        unlabeled_image_dataset_list, test_size=0.1, random_state=0
    )
    print("{:15}all: {}, train: {}, test: {}".format("", len(unlabeled_image_dataset_list), len(train_images), len(test_images)))

    # Load the datasets and mean file
    print("[ PREPROCESS ] Load the datasets and mean file.")
    mean = np.load(args.mean)
    train = PreprocessedDataset(train_images, args.root, mean, model.insize)
    val = PreprocessedDataset(test_images, args.root, mean, model.insize, False)

    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize, n_processes=args.loaderjob)
    val_iter = chainer.iterators.MultiprocessIterator(
        val, args.val_batchsize, repeat=False, n_processes=args.loaderjob)

    # Set up an optimizer
    print("[ PREPROCESS ] Set up an optimizer.")
    optimizer = chainer.optimizers.Adam()
    # optimizer = chainer.optimizers.MomentumSGD(lr=0.005, momentum=0.9)
    optimizer.setup(model)

    # Set up a trainer
    print("[ PREPROCESS ] Set up a trainer.")
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    val_interval = (10 if args.test else 1000), 'iteration'
    log_interval = (10 if args.test else 1000), 'iteration'

    trainer.extend(TestModeEvaluator(val_iter, model, device=args.gpu),
                   trigger=val_interval)

    trainer.extend(extensions.snapshot(), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'cae_model_iter_{.updater.iteration}'), trigger=val_interval)

    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss', 'lr', 'elapsed_time'
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.run()

    # Save the trained model
    serializers.save_npz(os.path.join(args.out, "cae_model_final.npz"), model)
    serializers.save_npz(os.path.join(args.out, "cae_optimaizer_final.npz"), optimizer)

    print("[ FINISH ] Training is Finished.")
