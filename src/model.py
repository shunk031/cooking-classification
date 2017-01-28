# -*- coding: utf-8 -*-

import math

import chainer
import chainer.functions as F
import chainer.links as L


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
            conv1=L.Convolution2D(None,  96, 11, stride=4),
            bn1=L.BatchNormalization(96),

            conv2=L.Convolution2D(None, 256,  5, pad=2),
            bn2=L.BatchNormalization(256),

            conv3=L.Convolution2D(None, 384,  3, pad=1),
            bn3=L.BatchNormalization(384),

            conv4=L.Convolution2D(None, 384,  3, pad=1),
            bn4=L.BatchNormalization(384),

            conv5=L.Convolution2D(None, 384,  3, pad=1),
            bn5=L.BatchNormalization(384),

            conv6=L.Convolution2D(None, 384,  3, pad=1),
            bn6=L.BatchNormalization(384),

            conv7=L.Convolution2D(None, 256,  3, pad=1),
            bn7=L.BatchNormalization(256),

            fc8=L.Linear(None, 4096),
            fc9=L.Linear(None, 4096),
            fc10=L.Linear(None, 25),
        )
        self.train = True

    def __call__(self, x, t):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.bn1(self.conv1(x)))), 3, stride=2)
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


class BottleNeckA(chainer.Chain):

    def __init__(self, in_size, ch, out_size, stride=2):
        w = math.sqrt(2)
        super(BottleNeckA, self).__init__(
            conv1=L.Convolution2D(in_size, ch, 1, stride, 0, w, nobias=True),
            bn1=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, w, nobias=True),
            bn2=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, out_size, 1, 1, 0, w, nobias=True),
            bn3=L.BatchNormalization(out_size),

            conv4=L.Convolution2D(in_size, out_size, 1, stride, 0, w, nobias=True),
            bn4=L.BatchNormalization(out_size),
        )

    def __call__(self, x, train):
        h1 = F.relu(self.bn1(self.conv1(x), test=not train))
        h1 = F.relu(self.bn2(self.conv2(h1), test=not train))
        h1 = self.bn3(self.conv3(h1), test=not train)
        h2 = self.bn4(self.conv4(x), test=not train)

        return F.relu(h1 + h2)


class BottleNeckB(chainer.Chain):

    def __init__(self, in_size, ch):
        w = math.sqrt(2)
        super(BottleNeckB, self).__init__(
            conv1=L.Convolution2D(in_size, ch, 1, 1, 0, w, nobias=True),
            bn1=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, w, nobias=True),
            bn2=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, in_size, 1, 1, 0, w, nobias=True),
            bn3=L.BatchNormalization(in_size),
        )

    def __call__(self, x, train):
        h = F.relu(self.bn1(self.conv1(x), test=not train))
        h = F.relu(self.bn2(self.conv2(h), test=not train))
        h = self.bn3(self.conv3(h), test=not train)

        return F.relu(h + x)


class Block(chainer.Chain):

    def __init__(self, layer, in_size, ch, out_size, stride=2):
        super(Block, self).__init__()
        links = [('a', BottleNeckA(in_size, ch, out_size, stride))]
        for i in range(layer - 1):
            links += [('b{}'.format(i + 1), BottleNeckB(out_size, ch))]

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

    def __init__(self):
        w = math.sqrt(2)
        super(ResNet, self).__init__(
            conv1=L.Convolution2D(3, 64, 7, 2, 3, w, nobias=True),
            bn1=L.BatchNormalization(64),
            res2=Block(3, 64, 64, 256, 1),
            res3=Block(4, 256, 128, 512),
            res4=Block(6, 512, 256, 1024),
            res5=Block(3, 1024, 512, 2048),
            fc=L.Linear(None, 25),
        )
        self.train = True

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x, t):
        self.clear()
        h = self.bn1(self.conv1(x), test=not self.train)
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h, self.train)
        h = self.res3(h, self.train)
        h = self.res4(h, self.train)
        h = self.res5(h, self.train)
        h = F.average_pooling_2d(h, 7, stride=1)
        h = self.fc(h)

        if self.train:
            self.loss = F.softmax_cross_entropy(h, t)
            self.accuracy = F.accuracy(h, t)
            chainer.report({'loss': self.loss, 'accuracy': self.accuracy}, self)
            return self.loss
        else:
            return h
