# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import Variable


class StackedCAE(chainer.Chain):

    insize = 227

    def __init__(self):
        super(StackedCAE, self).__init__(
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

            fc11=L.Linear(25, 4096),
            fc12=L.Linear(4096, 4096),
            fc13=L.Linear(4096, 9216),

            deconv14=L.Deconvolution2D(256, 384,  3, pad=1),
            bn14=L.BatchNormalization(384),

            deconv15=L.Deconvolution2D(384, 384,  3, pad=1),
            bn15=L.BatchNormalization(384),

            deconv16=L.Deconvolution2D(384, 384,  3, pad=1),
            bn16=L.BatchNormalization(384),

            deconv17=L.Deconvolution2D(384, 384,  3, pad=1),
            bn17=L.BatchNormalization(384),

            deconv18=L.Deconvolution2D(384, 256,  3, pad=1),
            bn18=L.BatchNormalization(256),

            deconv19=L.Deconvolution2D(256, 96,  5, pad=2),
            bn19=L.BatchNormalization(96),

            deconv20=L.Deconvolution2D(96, 3,  11, stride=4),
        )
        self.train = True

    def __call__(self, x, _):

        # encode
        self.z = self.encode(x)
        xp = cuda.get_array_module(self.z.data)
        volatile = 'off' if self.train else 'on'
        z_t = Variable(xp.ones_like(self.z.data), volatile=volatile)
        loss_z = F.mean_squared_error(self.z, z_t)

        # decode
        self.y = self.decode(self.z)
        loss_y = F.mean_squared_error(x, self.y)
        loss = loss_z + loss_y
        chainer.report({'loss': loss}, self)
        return loss

    def encode(self, x):

        h = self.conv1(x)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.local_response_normalization(h)
        self.pool1_inshape_ = h.data.shape
        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.conv2(h)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.local_response_normalization(h)
        self.pool2_inshape_ = h.data.shape
        h = F.max_pooling_2d(h, 3, stride=2)

        h = self.conv3(h)
        h = self.bn3(h)
        h = F.relu(h)

        h = self.conv4(h)
        h = self.bn4(h)
        h = F.relu(h)

        h = self.conv5(h)
        h = self.bn5(h)
        h = F.relu(h)

        h = self.conv6(h)
        h = self.bn6(h)
        h = F.relu(h)

        h = self.conv7(h)
        h = self.bn7(h)
        h = F.relu(h)
        self.pool3_inshape_ = h.data.shape
        h = F.max_pooling_2d(h, 3, stride=2)
        self.pool3_outshape_ = h.data.shape

        h = self.fc8(h)
        h = F.relu(h)

        h = self.fc9(h)
        h = F.relu(h)

        h = self.fc10(h)

        z = h

        return z

    def decode(self, z):
        h = z

        h = self.fc11(h)
        h = F.relu(h)

        h = self.fc12(h)
        h = F.relu(h)

        h = self.fc13(h)
        h = F.relu(h)

        h = F.reshape(h, self.pool3_outshape_)
        h = F.unpooling_2d(h, 3, stride=2, outsize=self.pool3_inshape_[-2:])
        h = self.deconv14(h)
        h = self.bn14(h)
        h = F.relu(h)

        h = self.deconv15(h)
        h = self.bn15(h)
        h = F.relu(h)

        h = self.deconv16(h)
        h = self.bn16(h)
        h = F.relu(h)

        h = self.deconv17(h)
        h = self.bn17(h)
        h = F.relu(h)

        h = self.deconv18(h)
        h = self.bn18(h)
        h = F.relu(h)

        h = F.unpooling_2d(h, 3, stride=2, outsize=self.pool2_inshape_[-2:])
        h = self.deconv19(h)
        h = self.bn19(h)
        h = F.relu(h)

        h = F.unpooling_2d(h, 3, stride=2, outsize=self.pool1_inshape_[-2:])
        h = self.deconv20(h)
        h = F.relu(h)
        h = F.sigmoid(h)

        y = h
        return y
