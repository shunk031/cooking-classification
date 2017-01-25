# -*- coding: utf-8 -*-

import os
import argparse
import random
import pickle

from sklearn.model_selection import train_test_split
import numpy as np

import chainer
from chainer import training
from chainer import serializers
from chainer.training import extensions

import model

# import hauth
# from slacker import Slacker
# import json

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath("__file__")), "../dataset")

archs = {
    'alex': model.AlexNet,
    'alexlike': model.AlexLikeNet,
}


class PreprocessedDataset(chainer.dataset.DatasetMixin):

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
    # parser.add_argument('val', help='Path to validation image-label list file')
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
    parser.add_argument('--mean', '-m', default='mean.npy',
                        help='Mean file (computed by compute_mean.py)')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--root', '-R', default='.',
                        help='Root directory path of image files')
    parser.add_argument('--val_batchsize', '-b', type=int, default=250,
                        help='Validation minibatch size')
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()

    model = archs[args.arch]()
    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make the GPU current
        model.to_gpu()

    print("[ PREPROCESS ] Load image-label list file.")
    with open(args.train, "rb") as rf:
        labeled_image_dataset_list = pickle.load(rf)

    print("[ PREPEOCESS ] Split train and test tuple.")
    train_tuples, test_tuples = train_test_split(
        labeled_image_dataset_list, test_size=0.3, random_state=0
    )

    # Load the datasets and mean file
    print("[ PREPROCESS ] Load the datasets and mean file.")
    mean = np.load(args.mean)
    train = PreprocessedDataset(train_tuples, args.root, mean, model.insize)
    val = PreprocessedDataset(test_tuples, args.root, mean, model.insize, False)
    # These iterators load the images with subprocesses running in parallel to
    # the training/validation.
    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize, n_processes=args.loaderjob)
    val_iter = chainer.iterators.MultiprocessIterator(
        val, args.val_batchsize, repeat=False, n_processes=args.loaderjob)

    # Set up an optimizer
    print("[ PREPROCESS ] Set up an optimizer.")
    # optimizer = chainer.optimizers.Adam()
    optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    # Set up a trainer
    print("[ PREPROCESS ] Set up a trainer.")
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    val_interval = (10 if args.test else 1000), 'iteration'
    log_interval = (10 if args.test else 1000), 'iteration'

    trainer.extend(TestModeEvaluator(val_iter, model, device=args.gpu),
                   trigger=val_interval)
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'), trigger=val_interval)

    # Be careful to pass the interval directly to LogReport
    # (it determines when to emit log rather than when to read observations)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy', 'lr'
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

    # Save the trained model
    serializers.save_npz(os.path.join(args.out, "model_final.npz"), model)
    serializers.save_npz(os.path.join(args.out, "optimaizer_final.npz"), optimizer)

    print("[ FINISH ] Training is Finished.")
