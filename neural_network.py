import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


from chainer.training import extensions
def augment(x, y):
    angles = np.random.choice([0, 1, 2, 3], len(x))
    for i in range(1,4):
        index = np.where(angles == i)[0]
        x[index] = np.rot90(x[index], i, (2, 3))
        if i%2 != 0:
            c = y[index, 3]
            y[index, 3] = y[index, 2]
            y[index, 2] = c
        theta = np.radians(90 * i)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        y[index, :2] = np.dot(y[index, :2], R)


class FaceNet():
    def __init__(self):
        self.model = Chain(
            conv1=L.Convolution2D(3, 20, 3, 1, 1),
            conv2=L.Convolution2D(20, 20, 3, 1, 1),

            conv3=L.Convolution2D(20, 40, 3, 1, 1),
            conv4=L.Convolution2D(40, 40, 3, 1, 1),

            linear1=L.Linear(None, 100),
            linear2=L.Linear(100, 4)
        )

        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

    def foward(self, x):
        out = self.model.conv1(x)
        out = F.elu(out)
        out = self.model.conv2(out)

        out = F.max_pooling_2d(out, 2)
        out = F.elu(out)
        out = self.model.conv3(out)
        out = F.elu(out)
        out = self.model.conv4(out)
        out = F.elu(out)

        out = F.average_pooling_2d(out, 6)
        out = F.dropout(out)
        out = self.model.linear1(out)
        out = F.elu(out)
        out = F.dropout(out)
        out = self.model.linear2(out)


        return out

    def predict(self, X, step=100):
        with chainer.using_config('train', False):
            with chainer.no_backprop_mode():
                output = []
                for i in range(0, len(X), step):
                    x = Variable(X[i:i + step])
                    output.append(self.foward(x).data)
                return np.vstack(output)

    def score(self, X, Y, step=100):
        predicted = self.predict(X, step)
        score = F.r2_score(predicted, Y).data
        return score



    def fit(self, X, Y, batchsize=100, n_epoch=10):
        with chainer.using_config('train', True):
            learning_curve = []
            for epoch in range(n_epoch):
                print('epoch ',epoch)
                index = np.random.permutation(len(X))
                for i in range(0, len(index), batchsize):
                    self.model.cleargrads()
                    print(i)
                    x = X[index[i:i+batchsize]]
                    y = Y[index[i:i+batchsize]]
                    #augment(x, y)

                    x = Variable(x)
                    y = Variable(y)

                    output = self.foward(x)
                    loss = F.mean_squared_error(y, output)
                    loss.backward()

                    learning_curve.append(float(loss.data))

                    self.optimizer.update()
            return learning_curve


