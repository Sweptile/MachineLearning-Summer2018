import numpy as np
import mxnet as mx
from mxnet import gluon, autograd, nd

ctx = mx.cpu()
mx.random.seed(1)

def binary_encode(i, digits):
    return np.array([i>>d&1 for d in range(digits)])
def binary_encode1(i, digits):
    return nd.array([i>>d&1 for d in range(digits)])

def fizzbuzz_encode(i):
    if i%15 == 0:
##        return np.array([0, 0, 0, 1])
        return 1
    elif i%5 == 0:
##        return np.array([0, 0, 1, 0])
        return 2
    elif i%3 == 0:
##        return np.array([0, 1, 0, 0])
        return 3
    else:
##        return np.array([1, 0, 0, 0])
        return 4

num_digits = 10

trX = np.array([binary_encode(i, num_digits) for i in range(101, 2**num_digits)])
trY = np.array([fizzbuzz_encode(i) for i in range(101, 2**num_digits)])

testX = np.array([binary_encode(i, num_digits) for i in range(1, 101)])
testX1 = nd.array([binary_encode(i, num_digits) for i in range(1, 101)])
testY = np.array([fizzbuzz_encode(i) for i in range(1, 101)])

hidden_layers = 1
hidden_units = 100
batch_size = 32

tr_data = mx.io.NDArrayIter(trX, trY, batch_size, shuffle=True)
##tr_data.reset()

##tr_data = gluon.data.DataLoader(gluon.data.ArrayDataset(trX, trY), batch_size=batch_size, shuffle=True)

net = gluon.nn.Sequential()
with net.name_scope():
##    net.add(gluon.nn.Dense(num_digits, activation="relu"))  #Input Layer
    net.add(gluon.nn.Dense(hidden_units, activation = "relu"))  #Hidden Layer
    net.add(gluon.nn.Dense(4))  #Output Layer

net.collect_params().initialize(mx.init.Normal(0.01), ctx=ctx)
loss_func = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.05})


epochs = 10000

for e in range(epochs):
    cumulative_loss = 0
    for i, batch in enumerate(tr_data):
        data = batch.data[0].as_in_context(ctx)
##        print(data)
        label = batch.label[0].as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = loss_func(output, label)
        loss.backward()
        trainer.step(data.shape[0])
        cumulative_loss += nd.sum(loss).asscalar()
##        cumulative_loss += loss

##    print(e, cumulative_loss/(1024-101))

a = nd.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])

##a = nd.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]])
##
##for i in testX1:
##    print()
print(net(a))
