import mxnet as mx
from mxnet import nd, gluon, autograd
import numpy as np

def label_encode(i):
    a = nd.zeros(10)
    a[i] = 1
    return a

ctx = mx.cpu(0)
mx.random.seed(1)
mnist = mx.test_utils.get_mnist()

batch_size = 32
tr_dataset = gluon.data.dataset.ArrayDataset(mnist["train_data"], nd.array(label_encode(i) for i in mnist["train_label"]))
test_dataset = gluon.data.dataset.ArrayDataset(mnist["test_data"], nd.array(label_encode(i) for i in mnist["test_label"]))

train_data = gluon.data.DataLoader(tr_dataset, batch_size, shuffle=True)
test_data = gluon.data.DataLoader(test_dataset, batch_size, shuffle=False)

hidden_layers = 1       #number of hidden layers
hidden_units = 100      #number of nodes in a hidden layer

net = gluon.nn.Sequential()                                         #making the neural net
with net.name_scope():
    net.add(gluon.nn.Dense(hidden_units, activation='relu'))        #hidden layer
    net.add(gluon.nn.Dense(10))                                      #output layer

net.collect_params().initialize(mx.init.Normal(sigma=0.01), ctx=ctx)                #setting the initial weights and biases
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=True)      #loss function (Softmax Cross Entropy)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.05})        #setting up the optimizer (Stochastic Gradient Descent)


epochs = 1

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
##
### Use Accuracy as the evaluation metric.
##metric = mx.metric.Accuracy()
### Reset the validation data iterator.
####test_data.reset()
### Loop over the validation data iterator.
##for i, (data, label) in enumerate(test_data):
##    # Splits validation data into multiple slices along batch_axis
##    # and copy each slice into a context.
####    data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
##    # Splits validation label into multiple slices along batch_axis
##    # and copy each slice into a context.
####    label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
##    data = data.as_in_context(ctx)
####    data = data.astype(np.float32)
##    label = label.as_in_context(ctx)
####    label = label.astype(np.float32)
##    outputs = []
##    for x in data:
##        outputs.append(net(x))
##    # Updates internal evaluation
##    print(outputs)
##    print(label)
##    metric.update(label, outputs)
##print('validation acc: %s=%f'%metric.get())
##assert metric.get()[1] > 0.94
##
##
##
##
##
####prediction = []
####correct = []
####        
####for i, (data, label) in enumerate(test_data):
####    data = data.as_in_context(ctx)
####    data = data.astype(np.float32)
####    for j in net(data):
####        prediction.append(fizzbuzz_decode(len(prediction)+1, nd.argmax(j, axis=0)))     #prediction array
####
####for i in prediction:
####    print(i)                                    #prints the final output
####
####for i, val in enumerate(testY):
####    correct.append(fizzbuzz_decode(i+1, np.argmax(val, axis=0)))
####
####accuracy = 0
####
####for i in range(100):
####    if prediction[i]==correct[i]:
####        accuracy+=1
####
####print('\nThe acuuracy of the training data is ' + str(accuracy) + '%')
