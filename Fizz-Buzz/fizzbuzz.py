import numpy as np
from mxnet import gluon, nd, autograd
import mxnet as mx

ctx = mx.cpu()
mx.random.seed(1)

def binary_encode(i, digits):
    return np.array([i>>d&1 for d in range(digits)])
def fizzbuzz_encode(i):
    if i%15 == 0:
        return np.array([0, 0, 0, 1])
    elif i%5 == 0:
        return np.array([0, 0, 1, 0])
    elif i%3 == 0:
        return np.array([0, 1, 0, 0])
    else:
        return np.array([1, 0, 0, 0])

def fizzbuzz_decode(i, pred):
    if pred == 0:
        return i
    elif pred == 1:
        return 'fizz'
    elif pred == 2:
        return 'buzz'
    else:
        return 'fizzbuzz'

num_digits = 10         #number of digits in the input

trX = np.array([binary_encode(i, num_digits) for i in range(101, 2**num_digits)])
trY = np.array([fizzbuzz_encode(i) for i in range(101, 2**num_digits)])
tr_dataset = gluon.data.dataset.ArrayDataset(trX, trY)          #training dataset

testX = np.array([binary_encode(i, num_digits) for i in range(1, 101)])
testY = np.array([fizzbuzz_encode(i) for i in range(1, 101)])
test_dataset = gluon.data.dataset.ArrayDataset(testX, testY)    #testing dataset

hidden_layers = 1       #number of hidden layers
hidden_units = 100      #number of nodes in a hidden layer
batch_size = 32

train_data = gluon.data.DataLoader(tr_dataset, batch_size, shuffle=True)
test_data = gluon.data.DataLoader(test_dataset, batch_size, shuffle=False)

net = gluon.nn.Sequential()                                         #making the neural net
with net.name_scope():
    net.add(gluon.nn.Dense(hidden_units, activation='relu'))        #hidden layer
    net.add(gluon.nn.Dense(4))                                      #output layer

net.collect_params().initialize(mx.init.Normal(sigma=0.01), ctx=ctx)                #setting the initial weights and biases
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)      #loss function (Softmax Cross Entropy)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.05})        #setting up the optimizer (Stochastic Gradient Descent)


epochs = 1000

for e in range(epochs):                             #training procedure
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx)
        data = data.astype(np.float32)
        label = label.as_in_context(ctx)
        label = label.astype(np.float32)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])
        cumulative_loss += nd.sum(loss).asscalar()

prediction = []
correct = []
        
for i, (data, label) in enumerate(test_data):
    data = data.as_in_context(ctx)
    data = data.astype(np.float32)
    for j in net(data):
        prediction.append(fizzbuzz_decode(len(prediction)+1, nd.argmax(j, axis=0)))     #prediction array

for i in prediction:
    print(i)                                    #prints the final output

for i, val in enumerate(testY):
    correct.append(fizzbuzz_decode(i+1, np.argmax(val, axis=0)))

accuracy = 0

for i in range(100):
    if prediction[i]==correct[i]:
        accuracy+=1

print('\nThe acuuracy of the training data is ' + str(accuracy) + '%')
