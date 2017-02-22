# -*- coding: utf-8 -*-

import os
import math

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.initializers import constant
from chainer.initializers import normal
from chainer.dataset import download
from chainer.serializers import npz


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
        h1 = self.bn3(self.conv3(h1), test=not train)
        h2 = self.bn4(self.conv4(x), test=not train)

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
        h = self.bn3(self.conv3(h), test=not train)

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
            _retrieve(
                'ResNet-50-model.npz', 'ResNet-50-model.caffemodel', self)
            self.fc6 = L.Linear(None, 25)
        elif pretrained_model:
            npz.load_npz(pretrained_model, self)

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
        chainermodel = cls(pretrained_model=None)
        _transfer_resnet50(caffemodel, chainermodel)
        npz.save_npz(path_npz, chainermodel, compression=False)

    def __call__(self, x, t):
        h = self.bn1(self.conv1(x), test=not self.train)
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h, self.train)
        h = self.res3(h, self.train)
        h = self.res4(h, self.train)
        h = self.res5(h, self.train)
        h = F.average_pooling_2d(h, 7, stride=1)
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
        h = F.average_pooling_2d(h, 7, stride=1)
        h = self.fc6(h)

        pred = F.softmax(h)
        return pred


def _make_npz(path_npz, path_caffemodel, model):
    print('Now loading caffemodel (usually it may take few minutes)')
    if not os.path.exists(path_caffemodel):
        raise IOError(
            'The pre-trained caffemodel does not exist. Please download it '
            'from \'https://github.com/KaimingHe/deep-residual-networks\', '
            'and place it on {}'.format(path_caffemodel))
    ResNet.convert_caffemodel_to_npz(path_caffemodel, path_npz)
    npz.load_npz(path_npz, model)
    return model


def _retrieve(name_npz, name_caffemodel, model):
    root = download.get_dataset_directory('pfnet/chainer/models/')
    path = os.path.join(root, name_npz)
    path_caffemodel = os.path.join(root, name_caffemodel)
    return download.cache_or_load_file(
        path, lambda path: _make_npz(path, path_caffemodel, model),
        lambda path: npz.load_npz(path, model))

archs = {
    "alex": AlexNet,
    "alexlike": AlexLikeNet,
    "deepalexlike": DeepAlexLikeNet,
    "resnet": ResNet,
}
