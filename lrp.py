# %load lrp.py
import chainer
from chainer.training import extensions
import argparse
import easydict
import numpy as np

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


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batch_size', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--output', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=500,
                        help='Number of units')
    args = parser.parse_args()
    # args = easydict.EasyDict(
    #     {
    #         "batch_size":100,
    #         "epoch":20,
    #         "gpu":0,
    #         "output":"result",
    #         "resume":False,
    #         "unit":1000,
    #         "frequency":1
    #     }
    # )
    model = chainer.links.Classifier(MLP(args.unit))

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train, test = chainer.datasets.get_mnist()

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

    y = model.predictor(np.array(map(lambda x: x[0], train[:20])))
    return y


def LRP(y):
    print y.shape
    v = y
    c = y.creator
    _a_ = []
    while(c is not None):
        print("\t c {}".format(c.label))
        for i in c.inputs:
            if not isinstance(i.data, type(None)):
                print("{} {}".format(i.name, i.data.shape))
        if not isinstance(c.inputs[0].data, type(None)):
            _a_.append(c.inputs)
        v = c.inputs[0]
        c = v.creator
    r = None
    x = None
    total_relevance = y.data.max(axis=1)
    print y.data.argmax(axis=1)
    for cnt, value in enumerate(_a_):
        i, w, b = value
        if cnt == 0:
            print "w data {} w data argmax {} total {}, i {}".format(
                w.data.shape, w.data[y.data.argmax(axis=1)].shape, total_relevance.shape, i.data.shape)
            print w.data[y.data.argmax(axis=1)]
            r = np.multiply(w.data[y.data.argmax(axis=1)], (total_relevance / y.data.max(axis=1))[0])* i.data
        else:
            print w.data.shape, x.data.shape
            tmp = np.zeros(w.data.T.shape)
            print "r : {}, tmp : {}, x : {}".format(r.shape, tmp.shape, x.data.shape)
            for k, _w in enumerate(w.data.T):
                tmp[k] = _w / (x.data[0] + 1e-5)
            r = i.data * np.dot(r, tmp.T)
        x = i
    return r


def LRP_v2(y):
    v = y
    c = y.creator
    _a_ = []
    while(c is not None):
        print("creator {}".format(c.label))
        for i in c.inputs:
            if not isinstance(i.data, type(None)):
                print("{} {}".format(i.name, i.data.shape))
        if not isinstance(c.inputs[0].data, type(None)):
            _a_.append(c.inputs)
        v = c.inputs[0]
        c = v.creator
    x = None
    r = np.zeros(y.data.shape)
    for i, _ in enumerate(y.data):
        r[i, _.argmax()] = _.max()
    print "r : {}".format(r.argmax(axis=1))
    for cnt, value in enumerate(_a_):
        x, w, b = value
        print "r : {}, x : {}, w : {}".format(r.shape, x.data.shape, w.data.shape)
        tmp = []
        for k, _w in enumerate(w.data):
            z = x.data * _w
            z = z / z.sum()
            tmp.append(z)
        tmp = np.asarray(tmp).transpose(1, 0, 2)
        print "A : {}".format(tmp.shape)
        r = np.asarray([np.dot(r[i, :], tmp[i, :]) for i in range(r.shape[0])])
    return r


if __name__ == '__main__':
    y = main()
    r = LRP(y)
    import matplotlib.pyplot as plt
    plt.figure()
    for i in r:
        plt.imshow(i.reshape(28, 28))
        plt.show()

    pass
