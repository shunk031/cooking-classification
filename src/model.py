# -*- coding: utf-8 -*-

import os
import math
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.initializers import constant
from chainer.initializers import normal
from chainer.dataset import download
from chainer.serializers import npz

from train_cae.autoencoder import StackedCAE


class AlexNet(chainer.Chain):

    insize = 227

    def __init__(self):
        super(AlexNet, self).__init__(
            conv1=L.Convolution2D(None,  96, 11, stride=4),
            conv2=L.Convolution2D(None, 256,  5, pad=2),
            conv3=L.Convolution2D(None, 384,  3, pad=1),
            conv4=L.Convolution2D(None, 384,  3, pad=1),
            conv5=L.Convolution2D(None, 256,  3, pad=1),
            fc6=L.Linear(None, 4096),
            fc7=L.Linear(None, 4096),
            fc8=L.Linear(None, 25),
        )
        self.train = True

    def __call__(self, x, t):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)), train=self.train)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train)
        h = self.fc8(h)

        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss


class AlexLikeNet(chainer.Chain):

    insize = 227

    def __init__(self):
        super(AlexLikeNet, self).__init__(
            conv1=L.Convolution2D(None,  96, 11, stride=4),
            bn1=L.BatchNormalization(96),

            conv2=L.Convolution2D(None, 256,  5, pad=2),
            bn2=L.BatchNormalization(256),

            conv3=L.Convolution2D(None, 384,  3, pad=1),
            bn3=L.BatchNormalization(384),

            conv4=L.Convolution2D(None, 384,  3, pad=1),
            bn4=L.BatchNormalization(384),

            conv5=L.Convolution2D(None, 256,  3, pad=1),
            bn5=L.BatchNormalization(256),

            fc6=L.Linear(None, 4096),
            fc7=L.Linear(None, 4096),
            fc8=L.Linear(None, 25),
        )
        self.train = True

    def __call__(self, x, t):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.bn1(self.conv1(x)))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.bn2(self.conv2(h)))), 3, stride=2)
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        h = F.max_pooling_2d(F.relu(self.bn5(self.conv5(h))), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)), train=self.train)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train)
        h = self.fc8(h)

        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss


class DeepAlexLikeNet(chainer.Chain):

    insize = 227

    def __init__(self):
        super(DeepAlexLikeNet, self).__init__(
            conv1=L.Convolution2D(3,  96, 11, stride=4),
            bn1=L.BatchNormalization(96),

            conv2=L.Convolution2D(96, 256,  5, pad=2),
            bn2=L.BatchNormalization(256),

            conv3=L.Convolution2D(256, 384,  3, pad=1),
            bn3=L.BatchNormalization(384),

            conv4=L.Convolution2D(384, 384,  3, pad=1),
            bn4=L.BatchNormalization(384),

            conv5=L.Convolution2D(384, 384,  3, pad=1),
            bn5=L.BatchNormalization(384),

            conv6=L.Convolution2D(384, 384,  3, pad=1),
            bn6=L.BatchNormalization(384),

            conv7=L.Convolution2D(384, 256,  3, pad=1),
            bn7=L.BatchNormalization(256),

            fc8=L.Linear(9216, 4096),
            fc9=L.Linear(4096, 4096),
            fc10=L.Linear(4096, 25),
        )
        self.train = True

    def __call__(self, x, t):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.bn2(self.conv2(h)))), 3, stride=2)
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        h = F.relu(self.bn5(self.conv5(h)))
        h = F.relu(self.bn6(self.conv6(h)))
        h = F.max_pooling_2d(F.relu(self.bn7(self.conv7(h))), 3, stride=2)
        h = F.dropout(F.relu(self.fc8(h)), train=self.train)
        h = F.dropout(F.relu(self.fc9(h)), train=self.train)
        h = self.fc10(h)

        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss

    def predict(self, x):

        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.bn2(self.conv2(h)))), 3, stride=2)
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        h = F.relu(self.bn5(self.conv5(h)))
        h = F.relu(self.bn6(self.conv6(h)))
        h = F.max_pooling_2d(F.relu(self.bn7(self.conv7(h))), 3, stride=2)
        h = F.dropout(F.relu(self.fc8(h)), train=self.train)
        h = F.dropout(F.relu(self.fc9(h)), train=self.train)
        h = self.fc10(h)

        pred = F.softmax(h)
        return pred


class BottleNeckA(chainer.Chain):

    def __init__(self, in_size, ch, out_size, stride=2, initialW=None):

        super(BottleNeckA, self).__init__(
            conv1=L.Convolution2D(in_size, ch, 1, stride, 0, initialW=initialW, nobias=True),
            bn1=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, initialW=initialW, nobias=True),
            bn2=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, out_size, 1, 1, 0, initialW=initialW, nobias=True),
            bn3=L.BatchNormalization(out_size),
            conv4=L.Convolution2D(in_size, out_size, 1, stride, 0, initialW=initialW, nobias=True),
            bn4=L.BatchNormalization(out_size),
        )

    def __call__(self, x, train):
        h1 = F.relu(self.bn1(self.conv1(x), test=not train))
        h1 = F.relu(self.bn2(self.conv2(h1), test=not train))
        # h1 = self.bn3(self.conv3(F.dropout(h1, train=train, ratio=.4)), test=not train)
        # h2 = self.bn4(self.conv4(F.dropout(x, train=train, ratio=.4)), test=not train)
        h1 = F.dropout(self.bn3(self.conv3(h1), test=not train), train=train, ratio=.4)
        h2 = F.dropout(self.bn4(self.conv4(x), test=not train), train=train, ratio=.4)

        return F.relu(h1 + h2)


class BottleNeckB(chainer.Chain):

    def __init__(self, in_size, ch, initialW=None):

        super(BottleNeckB, self).__init__(
            conv1=L.Convolution2D(in_size, ch, 1, 1, 0, initialW=initialW, nobias=True),
            bn1=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, initialW=initialW, nobias=True),
            bn2=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, in_size, 1, 1, 0, initialW=initialW, nobias=True),
            bn3=L.BatchNormalization(in_size),
        )

    def __call__(self, x, train):
        h = F.relu(self.bn1(self.conv1(x), test=not train))
        h = F.relu(self.bn2(self.conv2(h), test=not train))
        # h = self.bn3(self.conv3(F.dropout(h, train=train, ratio=.4)), test=not train)
        h = F.dropout(self.bn3(self.conv3(h), test=not train), train=train, ratio=.4)

        return F.relu(h + x)


class Block(chainer.Chain):

    def __init__(self, layer, in_size, ch, out_size, stride=2, initialW=None):
        super(Block, self).__init__()
        links = [('a', BottleNeckA(in_size, ch, out_size, stride, initialW))]
        for i in range(layer - 1):
            links += [('b{}'.format(i + 1), BottleNeckB(out_size, ch, initialW))]

        for l in links:
            self.add_link(*l)
        self.forward = links

    def __call__(self, x, train):
        for name, _ in self.forward:
            f = getattr(self, name)
            x = f(x, train)

        return x


class ResNet(chainer.Chain):

    insize = 227

    def __init__(self, pretrained_model="auto"):
        if pretrained_model:
            kwargs = {'initialW': constant.Zero()}
        else:
            kwargs = {'initialW': normal.HeNormal(scale=1.0)}

        super(ResNet, self).__init__(
            conv1=L.Convolution2D(3, 64, 7, 2, 3, **kwargs),
            bn1=L.BatchNormalization(64),
            res2=Block(3, 64, 64, 256, 1, **kwargs),
            res3=Block(4, 256, 128, 512, 2, **kwargs),
            res4=Block(6, 512, 256, 1024, 2, **kwargs),
            res5=Block(3, 1024, 512, 2048, 2, **kwargs),
            fc6=L.Linear(None, 1000),
        )
        if pretrained_model == 'auto':
            print("[ PREPROCESS ] Use caffe model of ResNet.")
            self._retrieve(
                'ResNet-50-model.npz', 'ResNet-50-model.caffemodel', self)
            self.fc6 = L.Linear(None, 25)
        elif pretrained_model:
            npz.load_npz(pretrained_model, self)

        self.train = True

    def convert_caffemodel_to_npz(self, path_caffemodel, path_npz):
        """Converts a pre-trained caffemodel to a chainer model.
        Args:
            path_caffemodel (str): Path of the pre-trained caffemodel.
            path_npz (str): Path of the converted chainer model.
        """

        # As CaffeFunction uses shortcut symbols,
        # we import CaffeFunction here.
        from chainer.links.caffe.caffe_function import CaffeFunction
        caffemodel = CaffeFunction(path_caffemodel)
        chainermodel = self(pretrained_model=None)
        _transfer_resnet50(caffemodel, chainermodel)
        npz.save_npz(path_npz, chainermodel, compression=False)

    def _global_average_pooling_2d(self, x):
        n, channel, rows, cols = x.data.shape
        h = F.average_pooling_2d(x, (rows, cols), stride=1)
        h = F.reshape(h, (n, channel))
        return h

    def _make_npz(self, path_npz, path_caffemodel, model):
        print('Now loading caffemodel (usually it may take few minutes)')
        if not os.path.exists(path_caffemodel):
            raise IOError(
                'The pre-trained caffemodel does not exist. Please download it '
                'from \'https://github.com/KaimingHe/deep-residual-networks\', '
                'and place it on {}'.format(path_caffemodel))
        self.convert_caffemodel_to_npz(path_caffemodel, path_npz)
        npz.load_npz(path_npz, model)
        return model

    def _retrieve(self, name_npz, name_caffemodel, model):
        print("[ PREPROCESS ] Retrieve.")
        root = download.get_dataset_directory('pfnet/chainer/models/')
        path = os.path.join(root, name_npz)
        path_caffemodel = os.path.join(root, name_caffemodel)
        return download.cache_or_load_file(
            path, lambda path: self._make_npz(path, path_caffemodel, model),
            lambda path: npz.load_npz(path, model))

    def __call__(self, x, t):
        h = self.bn1(self.conv1(x), test=not self.train)
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h, self.train)
        h = self.res3(h, self.train)
        h = self.res4(h, self.train)
        h = self.res5(h, self.train)
        h = self._global_average_pooling_2d(h)
        h = self.fc6(h)

        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss

    def predict(self, x):
        h = self.bn1(self.conv1(x), test=not self.train)
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h, self.train)
        h = self.res3(h, self.train)
        h = self.res4(h, self.train)
        h = self.res5(h, self.train)
        h = self._global_average_pooling_2d(h)
        h = self.fc6(h)

        pred = F.softmax(h)
        return pred


class ImpBottleNeckA(chainer.Chain):

    def __init__(self, in_size, ch, out_size, stride=2, initialW=None):

        super(ImpBottleNeckA, self).__init__(
            conv1=L.Convolution2D(in_size, ch, 1, stride, 0, initialW=initialW, nobias=True),
            bn1=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, initialW=initialW, nobias=True),
            bn2=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, out_size, 1, 1, 0, initialW=initialW, nobias=True),
            bn3=L.BatchNormalization(out_size),
            conv4=L.Convolution2D(in_size, out_size, 1, stride, 0, initialW=initialW, nobias=True),
            bn4=L.BatchNormalization(out_size),
        )

    def __call__(self, x, train):
        h1 = F.leaky_relu(self.bn1(self.conv1(x), test=not train))
        h1 = F.leaky_relu(self.bn2(self.conv2(h1), test=not train))

        # Basic
        h1 = self.bn3(self.conv3(h1), test=not train)
        h2 = self.bn4(self.conv4(x), test=not train)

        # Conv -> BN -> Dropout
        # h1 = F.dropout(self.bn3(self.conv3(h1), test=not train), train=train, ratio=0.1)
        # h2 = F.dropout(self.bn4(self.conv4(x), test=not train), train=train, ratio=0.1)

        # Dropout -> Conv -> BN
        # h1 = self.bn3(self.conv3(F.dropout(h1, train=train, ratio=0.2)), test=not train)
        # h2 = self.bn4(self.conv4(F.dropout(x, train=train, ratio=0.2)), test=not train)

        return F.leaky_relu(h1 + h2)


class ImpBottleNeckB(chainer.Chain):

    def __init__(self, in_size, ch, initialW=None):

        super(ImpBottleNeckB, self).__init__(
            conv1=L.Convolution2D(in_size, ch, 1, 1, 0, initialW=initialW, nobias=True),
            bn1=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, initialW=initialW, nobias=True),
            bn2=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, in_size, 1, 1, 0, initialW=initialW, nobias=True),
            bn3=L.BatchNormalization(in_size),
        )

    def __call__(self, x, train):
        h = F.leaky_relu(self.bn1(self.conv1(x), test=not train))
        h = F.leaky_relu(self.bn2(self.conv2(h), test=not train))

        # Basic
        h = self.bn3(self.conv3(h), test=not train)

        # Conv -> BN -> Dropout
        # h = F.dropout(self.bn3(self.conv3(h), test=not train), train=train, ratio=0.1)

        # Dropout -> Conv -> BN
        # h = self.bn3(self.conv3(F.dropout(h, train=train, ratio=0.2)), test=not train)

        return F.leaky_relu(h + x)


class ImpBlock(chainer.Chain):

    def __init__(self, layer, in_size, ch, out_size, stride=2, initialW=None):
        super(ImpBlock, self).__init__()
        links = [('a', ImpBottleNeckA(in_size, ch, out_size, stride, initialW))]
        for i in range(layer - 1):
            links += [('b{}'.format(i + 1), ImpBottleNeckB(out_size, ch, initialW))]

        for l in links:
            self.add_link(*l)
        self.forward = links

    def __call__(self, x, train):
        for name, _ in self.forward:
            f = getattr(self, name)
            x = f(x, train)

        return x


class ImpResNet(chainer.Chain):

    insize = 227

    def __init__(self, pretrained_resnet="auto", pretrained_cae=False):
        if pretrained_resnet:
            kwargs = {'initialW': constant.Zero()}
        else:
            kwargs = {'initialW': normal.HeNormal(scale=1.0)}

        super(ImpResNet, self).__init__(
            conv1=L.Convolution2D(3, 64, 7, 2, 3, **kwargs),
            bn1=L.BatchNormalization(64),
            res2=ImpBlock(3, 64, 64, 256, 1, **kwargs),
            res3=ImpBlock(4, 256, 128, 512, 2, **kwargs),
            res4=ImpBlock(6, 512, 256, 1024, 2, **kwargs),
            res5=ImpBlock(3, 1024, 512, 2048, 2, **kwargs),
            fc6=L.Linear(None, 1000),
        )

        if pretrained_resnet == 'auto':
            print("[ PREPROCESS ] Use caffe model of ResNet.")
            self._retrieve('ResNet-50-model.npz', 'ResNet-50-model.caffemodel', self)

        elif pretrained_resnet:
            npz.load_npz(pretrained_resnet, self)

        cae_model = StackedCAE()

        if pretrained_cae == 'auto':
            print("[ PREPORCESS ] Use pre-trained CAE full connect layer.")

            cae_model_path = os.path.join(".", "train_cae", "result", "cae_model_final.npz")
            npz.load_npz(cae_model_path, cae_model)
        elif pretrained_cae:
            cae_model_path = pretrained_cae
            npz.load_npz(cae_model_path, cae_model)

        # change fc6 Layer beacause of 25 class classification
        del self.__dict__["fc6"]
        self._children.remove("fc6")
        if pretrained_cae:
            cae_linear = cae_model["fc10"].copy()
            self.add_link("fc6", cae_linear)
        else:
            self.add_link("fc6", L.Linear(None, 25))

        self.train = True

    def convert_caffemodel_to_npz(self, path_caffemodel, path_npz):
        """Converts a pre-trained caffemodel to a chainer model.
        Args:
            path_caffemodel (str): Path of the pre-trained caffemodel.
            path_npz (str): Path of the converted chainer model.
        """

        # As CaffeFunction uses shortcut symbols,
        # we import CaffeFunction here.
        from chainer.links.caffe.caffe_function import CaffeFunction
        caffemodel = CaffeFunction(path_caffemodel)
        chainermodel = self(pretrained_resnet=None)
        _transfer_resnet50(caffemodel, chainermodel)
        npz.save_npz(path_npz, chainermodel, compression=False)

    def _make_npz(self, path_npz, path_caffemodel, model):
        print('Now loading caffemodel (usually it may take few minutes)')
        if not os.path.exists(path_caffemodel):
            raise IOError(
                'The pre-trained caffemodel does not exist. Please download it '
                'from \'https://github.com/KaimingHe/deep-residual-networks\', '
                'and place it on {}'.format(path_caffemodel))
        self.convert_caffemodel_to_npz(path_caffemodel, path_npz)
        npz.load_npz(path_npz, model)
        return model

    def _retrieve(self, name_npz, name_caffemodel, model):
        print("[ PREPROCESS ] Called ImpResNet#_retrieve, {}".format(model.__class__.__name__))
        root = download.get_dataset_directory('pfnet/chainer/models/')
        path = os.path.join(root, name_npz)
        path_caffemodel = os.path.join(root, name_caffemodel)
        return download.cache_or_load_file(
            path, lambda path: self._make_npz(path, path_caffemodel, model),
            lambda path: npz.load_npz(path, model))

    def __call__(self, x, t):
        h = self.bn1(self.conv1(x), test=not self.train)
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h, self.train)
        h = self.res3(h, self.train)
        h = self.res4(h, self.train)
        h = self.res5(h, self.train)
        h = self._global_average_pooling_2d(h)
        h = self.fc6(h)

        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss

    def _global_average_pooling_2d(self, x):
        n, channel, rows, cols = x.data.shape
        h = F.average_pooling_2d(x, (rows, cols), stride=1)
        h = F.reshape(h, (n, channel))
        return h

    def predict(self, x):
        h = self.bn1(self.conv1(x), test=not self.train)
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h, self.train)
        h = self.res3(h, self.train)
        h = self.res4(h, self.train)
        h = self.res5(h, self.train)
        h = self._global_average_pooling_2d(h)
        h = self.fc6(h)

        pred = F.softmax(h)
        return pred


class ImpResNet101(chainer.Chain):

    insize = 227

    def __init__(self, pretrained_resnet="auto", pretrained_cae=False):
        if pretrained_resnet:
            kwargs = {'initialW': constant.Zero()}
        else:
            kwargs = {'initialW': normal.HeNormal(scale=1.0)}

        super(ImpResNet101, self).__init__(
            conv1=L.Convolution2D(3, 64, 7, 2, 3, **kwargs),
            bn1=L.BatchNormalization(64),
            res2=ImpBlock(3, 64, 64, 256, 1, **kwargs),
            res3=ImpBlock(4, 256, 128, 512, 2, **kwargs),
            res4=ImpBlock(23, 512, 256, 1024, 2, **kwargs),
            res5=ImpBlock(3, 1024, 512, 2048, 2, **kwargs),
            fc6=L.Linear(2048, 1000),
        )

        if pretrained_resnet == 'auto':
            print("[ PREPROCESS ] Use caffe model of ResNet101.")
            self._retrieve('ResNet-101-model.npz', 'ResNet-101-model.caffemodel', self)

        elif pretrained_resnet:
            npz.load_npz(pretrained_resnet, self)

        cae_model = StackedCAE()

        if pretrained_cae == 'auto':
            print("[ PREPORCESS ] Use pre-trained CAE full connect layer.")

            cae_model_path = os.path.join(".", "train_cae", "result", "cae_model_final.npz")
            npz.load_npz(cae_model_path, cae_model)
        elif pretrained_cae:
            cae_model_path = pretrained_cae
            npz.load_npz(cae_model_path, cae_model)

        # change fc6 Layer beacause of 25 class classification
        del self.__dict__["fc6"]
        self._children.remove("fc6")
        if pretrained_cae:
            cae_linear = cae_model["fc10"].copy()
            self.add_link("fc6", cae_linear)
        else:
            self.add_link("fc6", L.Linear(None, 25))

        self.train = True

    @classmethod
    def convert_caffemodel_to_npz(cls, path_caffemodel, path_npz):
        """Converts a pre-trained caffemodel to a chainer model.
        Args:
            path_caffemodel (str): Path of the pre-trained caffemodel.
            path_npz (str): Path of the converted chainer model.
        """

        # As CaffeFunction uses shortcut symbols,
        # we import CaffeFunction here.
        from chainer.links.caffe.caffe_function import CaffeFunction
        caffemodel = CaffeFunction(path_caffemodel)
        chainermodel = cls(pretrained_resnet=None)
        _transfer_resnet101(caffemodel, chainermodel)
        npz.save_npz(path_npz, chainermodel, compression=False)

    def _make_npz(self, path_npz, path_caffemodel, model):
        print('Now loading caffemodel (usually it may take few minutes)')
        if not os.path.exists(path_caffemodel):
            raise IOError(
                'The pre-trained caffemodel does not exist. Please download it '
                'from \'https://github.com/KaimingHe/deep-residual-networks\', '
                'and place it on {}'.format(path_caffemodel))
        ImpResNet101.convert_caffemodel_to_npz(path_caffemodel, path_npz)
        npz.load_npz(path_npz, model)
        return model

    def _retrieve(self, name_npz, name_caffemodel, model):
        print("[ PREPROCESS ] Called ImpResNet101#_retrieve, {}".format(model.__class__.__name__))
        root = download.get_dataset_directory('pfnet/chainer/models/')
        path = os.path.join(root, name_npz)
        path_caffemodel = os.path.join(root, name_caffemodel)
        return download.cache_or_load_file(
            path, lambda path: self._make_npz(path, path_caffemodel, model),
            lambda path: npz.load_npz(path, model))

    def __call__(self, x, t):
        h = self.bn1(self.conv1(x), test=not self.train)
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h, self.train)
        h = self.res3(h, self.train)
        h = self.res4(h, self.train)
        h = self.res5(h, self.train)
        h = self._global_average_pooling_2d(h)
        h = self.fc6(h)

        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss

    def _global_average_pooling_2d(self, x):
        n, channel, rows, cols = x.data.shape
        h = F.average_pooling_2d(x, (rows, cols), stride=1)
        h = F.reshape(h, (n, channel))
        return h

    def predict(self, x):
        h = self.bn1(self.conv1(x), test=not self.train)
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h, self.train)
        h = self.res3(h, self.train)
        h = self.res4(h, self.train)
        h = self.res5(h, self.train)
        h = self._global_average_pooling_2d(h)
        h = self.fc6(h)

        pred = F.softmax(h)
        return pred


def _transfer_components(src, dst_conv, dst_bn, bname, cname):
    src_conv = getattr(src, 'res{}_branch{}'.format(bname, cname))
    src_bn = getattr(src, 'bn{}_branch{}'.format(bname, cname))
    src_scale = getattr(src, 'scale{}_branch{}'.format(bname, cname))
    dst_conv.W.data[:] = src_conv.W.data
    dst_bn.avg_mean[:] = src_bn.avg_mean
    dst_bn.avg_var[:] = src_bn.avg_var
    dst_bn.gamma.data[:] = src_scale.W.data
    dst_bn.beta.data[:] = src_scale.bias.b.data


def _transfer_bottleneckA(src, dst, name):
    _transfer_components(src, dst.conv1, dst.bn1, name, '2a')
    _transfer_components(src, dst.conv2, dst.bn2, name, '2b')
    _transfer_components(src, dst.conv3, dst.bn3, name, '2c')
    _transfer_components(src, dst.conv4, dst.bn4, name, '1')


def _transfer_bottleneckB(src, dst, name):
    _transfer_components(src, dst.conv1, dst.bn1, name, '2a')
    _transfer_components(src, dst.conv2, dst.bn2, name, '2b')
    _transfer_components(src, dst.conv3, dst.bn3, name, '2c')


def _transfer_block(src, dst, names):
    _transfer_bottleneckA(src, dst.a, names[0])
    for i, name in enumerate(names[1:]):
        dst_bottleneckB = getattr(dst, 'b{}'.format(i + 1))
        _transfer_bottleneckB(src, dst_bottleneckB, name)


def _transfer_resnet50(src, dst):
    print("[ PREPROCESS ] Transfer ResNet 50.")
    dst.conv1.W.data[:] = src.conv1.W.data
    dst.conv1.b.data[:] = src.conv1.b.data
    dst.bn1.avg_mean[:] = src.bn_conv1.avg_mean
    dst.bn1.avg_var[:] = src.bn_conv1.avg_var
    dst.bn1.gamma.data[:] = src.scale_conv1.W.data
    dst.bn1.beta.data[:] = src.scale_conv1.bias.b.data

    _transfer_block(src, dst.res2, ['2a', '2b', '2c'])
    _transfer_block(src, dst.res3, ['3a', '3b', '3c', '3d'])
    _transfer_block(src, dst.res4, ['4a', '4b', '4c', '4d', '4e', '4f'])
    _transfer_block(src, dst.res5, ['5a', '5b', '5c'])

    dst.fc6.W.data[:] = src.fc1000.W.data
    dst.fc6.b.data[:] = src.fc1000.b.data


def _transfer_resnet101(src, dst):
    print("[ PREPROCESS ] Transfer ResNet 101.")
    dst.conv1.W.data[:] = src.conv1.W.data
    dst.bn1.avg_mean[:] = src.bn_conv1.avg_mean
    dst.bn1.avg_var[:] = src.bn_conv1.avg_var
    dst.bn1.gamma.data[:] = src.scale_conv1.W.data
    dst.bn1.beta.data[:] = src.scale_conv1.bias.b.data

    _transfer_block(src, dst.res2, ['2a', '2b', '2c'])
    _transfer_block(src, dst.res3, ['3a', '3b1', '3b2', '3b3'])
    _transfer_block(src, dst.res4, ['4a', '4b1', '4b2', '4b3', '4b4', '4b5', '4b6', '4b7', '4b8', '4b9', '4b10', '4b11',
                                    '4b12', '4b13', '4b14', '4b15', '4b16', '4b17', '4b18', '4b19', '4b20', '4b21', '4b22'])
    _transfer_block(src, dst.res5, ['5a', '5b', '5c'])

archs = {
    "alex": AlexNet,
    "alexlike": AlexLikeNet,
    "deepalexlike": DeepAlexLikeNet,
    "resnet": ResNet,
    "impresnet": ImpResNet,
    "impresnet101": ImpResNet101,
}
