import chainer
from chainer.training import extensions
import argparse
import numpy as np
from chainer.function import FunctionHook


class RetainOutputHook(FunctionHook):
    def forward_postprocess(self, function, in_data):
        function.retain_outputs([0])


class MLP(chainer.Chain):
    def __init__(self, n_units):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = chainer.links.Linear(None, n_units)
            self.l2 = chainer.links.Linear(None, n_units//2)
            self.l3 = chainer.links.Linear(None, 10)

    def __call__(self, x):
        x = chainer.functions.relu(self.l1(x))
        x = chainer.functions.relu(self.l2(x))
        return self.l3(x)


class CNN(chainer.Chain):
    def __init__(self, n_units):
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv1 = chainer.links.Convolution2D(in_channels=1, out_channels=n_units//2, ksize=3, stride=1)
            self.conv2 = chainer.links.Convolution2D(in_channels=None, out_channels=n_units, ksize=3, stride=1)
            self.l = chainer.links.Linear(None, 10)

    def __call__(self, x):
        x = chainer.functions.relu(self.conv1(x))
        x = chainer.functions.max_pooling_2d(x, ksize=2, stride=2)
        x = chainer.functions.relu(self.conv2(x))
        x = chainer.functions.max_pooling_2d(x, ksize=2, stride=2)
        return self.l(x)


def main(args):
    if args.net == "MLP":
        Net = MLP
        args.ndim = 1

    elif args.net == "CNN":
        Net = CNN
        args.ndim = 3

    model = chainer.links.Classifier(Net(args.unit))

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train, test = chainer.datasets.get_mnist(ndim=args.ndim)

    train_iter = chainer.iterators.SerialIterator(train, args.batch_size)
    test_iter = chainer.iterators.SerialIterator(test, args.batch_size, repeat=False, shuffle=False)

    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out=args.output)

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.extend(extensions.snapshot(), trigger=(args.frequency, 'epoch'))

    trainer.extend(extensions.LogReport())

    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'
            )
        )
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'
            )
        )

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']
    ))
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

    model.to_cpu()
    with RetainOutputHook():
        y = model.predictor(np.array(map(lambda x: x[0], train[:20])))
    return y


def LRP(z, epsilon=0):
    creator = z.creator
    var = z
    # relevance value
    r = np.zeros(z.data.shape)
    for i, d in enumerate(z.data):
        r[i, d.argmax()] = d.max()

    while(creator is not None):
        print "{}".format(creator.label)
        print "r:{}".format(r.shape)
        x = creator.inputs[0].data
        y = creator.outputs[0]().data

        if len(creator.inputs) > 1:
            w = creator.inputs[1].data
        if creator.label == "LinearFunction":
            print " x:{}\n w:{}\n y:{}\n".format(x.shape, w.shape, y.shape)
            _y = y + epsilon*np.sign(y)
            r = x.reshape(r.shape[0], -1) * (np.dot(r/_y, w))
            print " r {}".format(r.shape)
        elif creator.label == "Convolution2DFunction":
            print " x:{}\n w:{}\n y:{}\n".format(x.shape, w.shape, y.shape)
            _y = y + epsilon*np.sign(y)
            r = x * chainer.functions.deconvolution_2d(r.reshape(y.shape)/_y, w).data
            print " r {}".format(r.shape)
        elif creator.label == "MaxPooling2D":
            print " x:{}\n y:{}\n".format(x.shape, y.shape)
            r = chainer.functions.unpooling_2d(
                r.reshape(y.shape), ksize=creator.kh, stride=creator.sy, outsize=x.shape[2:]).data

        var = creator.inputs[0]
        creator = var.creator
    return r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batch_size', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--epsilon', '-epsilon', type=float, default=0.,
                        help='Value of epsilon')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--output', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=100,
                        help='Number of units')
    parser.add_argument('--net', '-n', default="CNN",
                        help='Name of network')
    parser.add_argument('--ndim', '-nd', type=int, default=3,
                        help='Dimensions of inputs')
    args = parser.parse_args()
    # import easydict
    # args = easydict.EasyDict(
    #     {
    #         "batch_size":100,
    #         "epoch":20,
    #         "epsilon":0,
    #         "gpu":0,
    #         "output":"result",
    #         "resume":False,
    #         "unit":100,
    #         "frequency":1,
    #         "net":"MLP",
    #         "ndim":1
    #     }
    # )

    y = main(args)
    res = LRP(y, epsilon=args.epsilon)
    c = y.creator
    v = y
    while(c is not None):
        v = c.inputs[0]
        c = v.creator
    x = v.data

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(
        res.shape[0], 2, figsize=(2*2, res.shape[0]*2), subplot_kw={'xticks': [], 'yticks': []})
    for i in range(res.shape[0]):
        im = x[i][0]
        axs[i, 0].imshow(im, vmin=im.min(), vmax=im.max(), cmap='gray')
        im = res[i][0]
        axs[i, 1].imshow(im, vmin=im.min(), vmax=im.max(), cmap='plasma')
    plt.savefig('lrp.png')
    plt.show()
