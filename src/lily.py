import random
import time
import pickle
import numpy as np

from math import log2

import matplotlib.pyplot as plt
from tqdm import trange

import requests, gzip
from pathlib import Path

class utils(object):
  def __init__(self):
    super(utils, self).__init__()
    self.grads, self.params, self.cache = {}, {}, {}

  def actFunc(self, input, deriv=False):
    ### relu
    #if deriv: return 1. * (input > 0)
    #return np.maximum(input, 0)

    ### leaky relu
    if deriv: return 1. * (input > 0.01)
    return np.maximum(0.01*input, input)

  def stable_softmax(self, X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps+0.001)

  def lossFunc(self, pred, y):
    #loss = -np.sum(y * np.log(pred))   # Cross Entropy Loss
    out = self.stable_softmax(pred) 
    dout = np.argmax(pred) - y
    loss = -sum([out[i]*log2(y+0.001) for i in range(len(out))])
    loss = loss[0]
    #loss = -sum([y*log2(out[i]+0.001) for i in range(len(out))])
    return loss, dout


class ANN(utils):
  def __init__(self, size):
    super(ANN, self).__init__()
    self.num = len(size)
    self.size = size

    # weights and bias through kaiming/he initialization
    dtype=np.float32  # data type
    self.params["b"] = [ np.random.uniform(-1.,1., size=(h, 1)).astype(dtype)/np.sqrt(2/h)   for h in size[1:] ]
    self.params["w"] = [ np.random.uniform(-1.,1., size=(w, h)).astype(dtype)/np.sqrt(2/h*w) for h, w in zip(size[:-1], size[1:]) ]

    # gradients for learning
    self.grads["b"]  = [ np.zeros(b.shape) for b in self.params['b'] ]
    self.grads["w"]  = [ np.zeros(w.shape) for w in self.params['w'] ]

  def forward(self, input):
    for weights, bias in zip(self.params["w"], self.params["b"]):
      input = self.actFunc( np.dot(weights, input) + bias )
    return input

  def backward(self, input, target):
    layer_vec = []                # each layer's z vector
    layer_out = []                # each layer's z vector after activation function
      
    nput = input * 1/255         # normlizing data
    #input = (input-min(input))/(max(input)-min(input))   #Normalized Data

    # determinig the output of each layer
    layer_out.append( input )     # adding the input to update first layer
    for weights, bias in zip(self.params["w"], self.params["b"]):
      input = np.dot(weights, input) + bias
      layer_vec.append(input)

      input = self.actFunc(input) 
      layer_out.append(input)

    output = np.argmax(input)
    #loss = int(output - target)  # loss function
    #loss = self.calc_loss(input, target)
    loss, dout = self.lossFunc(input, target)

    delta = dout * self.actFunc(layer_vec[-1], deriv=True)  
    self.grads["b"][-1] += delta
    self.grads["w"][-1] += np.dot(delta, layer_out[-2].T)

    # determing HIDDEN neurons weights and bias
    for l in range(2, self.num):
      delta = np.dot(self.params["w"][-l+1].T, delta) * self.actFunc( layer_vec[-l], True)  #  delta * dz/da * input 
      self.grads["b"][-l] += delta
      self.grads["w"][-l] += np.dot(delta, layer_out[-l-1].T)
    return loss, output

  #def clip(self, x): return 1000*(abs(x)> 1000) + x*(abs(x)<1000)
  def clip(self, x): return 1

  def optimize(self, lr, batch_size):
    #### SGD update rule
    self.params["b"] = [b-(lr/batch_size)* self.clip(nb) for b, nb in zip(self.params["b"], self.grads["b"])]
    self.params["w"] = [w-(lr/batch_size)* self.clip(nw) for w, nw in zip(self.params["w"], self.grads["w"])]
    print( self.grads["w"][0].mean() )

    ### clean grads
    self.grads["b"] = [ np.zeros(b.shape) for b in self.params['b'] ]
    self.grads["w"] = [ np.zeros(w.shape) for w in self.params['w'] ]

  def load(self, name):
    #print('loading params')
    with open(name + '.pickle', 'rb') as handle:
        self.params = pickle.load(handle)
    pass

  def save(self, name):
    #print('saving params')
    with open(name + '.pickle', 'wb') as handle:
        pickle.dump(self.params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pass


  #----------------------------------------------------------------------------------------------
# https://ludius0.github.io/my-blog/ai/deep%20learning%20(dl)/2020/12/14/Neural-network-from-scratch.html
#collapse-hide
#hide-output

def fetch(url):    
    name = url.split("/")[-1]
    dirs = Path("dataset/mnist")
    path = (dirs / name)
    if path.exists():
      with path.open("rb") as f:
        dat = f.read()
    else:
      if not dirs.is_dir(): dirs.mkdir(parents=True, exist_ok=True)
      with path.open("wb") as f:
        dat = requests.get(url).content
        f.write(dat)
    return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()

def mnist_dataset():
    print(" collecting data")
    X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
    X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]
    return (X_train, Y_train, X_test, Y_test)

X_train, Y_train, X_test, Y_test = mnist_dataset()


length = 100  #10000
images = X_train[:length].reshape((length,784,1))
labels = Y_train[:length]

net = ANN( [784,128,10] ) 
batch_size = 128
losses, accuracies = [], []
epoch = 1000 #10000
for e in (t := trange(1,epoch+1)):
  # Batch of training data & target data
  acc, count = 0, 0
  for i in range(0, len(images), batch_size):
    for img, tag in zip(images[i:i+batch_size], labels[i:i+batch_size]):
      loss, out = net.backward(img, tag)

      # Save for statistic
      #out = np.argmax(out) # results from Net
      #print(out, tag, out==tag)
      #acc = (out == tag).mean()

      count += (out == tag)
      t.set_description(f"out: {out:.0f}; tag: {tag:.1f}; state: {out==tag:.0f} loss: {loss:.5f}")
    acc, count = (acc + count/len(images))/2, 0

    accuracies.append(acc)
    losses.append(loss)
    
    #t.set_description(f"Loss: {loss:.5f}; Acc: {acc:.5f}")
    net.optimize(lr=0.9, batch_size = batch_size)
    #net.save("lily2")
    
plt.ylim(-0.01, 1.1)
plt.plot(losses)
plt.plot(accuracies)
plt.legend(["losses", "accuracies"])
plt.show()
# Evaluation


count = 0
X_test = X_test.reshape((len(X_test),784,1))
for img, tag in zip(X_test, Y_test):
  out = net.forward(img)
  preds = np.argmax(out, axis=0)
  if (tag == preds): count+=1
  print(f"Accuracy on testing set: {100*count/len(X_test)} %" , end='\r')


