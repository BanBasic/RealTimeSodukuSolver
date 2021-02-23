import time
import pickle
import numpy as np

import matplotlib.pyplot as plt
from tqdm import trange

from src.MNISTLoader import *
from src.rose import *

class cnn: # cnn_digit_recognizer
  def __init__(self, batch_size): 
    #           C, F, HH=WW, stride, pad
    self.model = [
      Convolution(1 , 32, 5, 1, 0), 
      #BatchNorm2d((batch_size*24*24,32)) ,          
      MaxPooling(2), 
      ReLU(),

      Convolution(32, 64, 5, 1, 0), 
      #BatchNorm2d((batch_size*8*8,64)) ,          
      MaxPooling(2),
      ReLU(),
      Linear(1024, 10, flatten=True),
      #ReLU(),
    ]

    # Loss function 
    self.soft_loss = SoftmaxCrossEntropy()
    pass

  def forward(self, input):
    for layer in self.model:
      input = layer.forward(input)
    return input

  def backward(self, input):
    for layer in reversed(self.model):
      input = layer.backward(input)
    pass

  # save weights and biases
  def save(self, name):  
    params = {}
    for i, layer in enumerate(self.model):
      if (type(layer) == Convolution or type(layer) == Linear ):
        params[i] = layer.params
    with open(name, 'wb') as handle:
      pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pass

  # load weights and biases
  def load(self, name):
    with open(name, 'rb') as handle:
        self.params = pickle.load(handle)
    for i, layer in enumerate(self.model):
      if (type(layer) == Convolution or type(layer) == Linear ):
        layer.params = params[i] 
    pass

  def optimize(self, lr, batch_size):
    for layer in self.model:
      if (type(layer) == Convolution or type(layer) == Linear ):
        # SGD
        layer.params["b"] -= (lr/batch_size) * layer.grads["b"]
        layer.params["w"] -= (lr/batch_size) * layer.grads["w"]

        #cleaning gradients
        layer.grads["b"] = np.zeros(layer.grads["b"].shape)
        layer.grads["w"] = np.zeros(layer.grads["w"].shape)
    pass

############# DATA
X_train, Y_train, X_test, Y_test = mnist_dataset()

############# Training
epoch = 10
batch_size = 10
losses, accuracies = [], []

amount = 25 #len(X_train) # amount of images to train 
############# CNN
net = cnn(batch_size)  

accuracy = 0
for e in (t := trange(1,epoch+1)):
  for i in range(0, amount, batch_size): # creating batchs
    images = np.array([ images for images in X_train[i:i+batch_size] ])
    labels = np.array([ labels for labels in Y_train[i:i+batch_size] ])

    #  model training
    preds = net.forward(images)
    preds = net.soft_loss.forward(preds)               
    dout, loss = net.soft_loss.backward(preds, labels)               
    net.backward(dout) 

    #  model stats 
    preds  = np.argmax(preds, axis=1) 
    labels = np.argmax(labels, axis=1)
    correct = (preds == labels).sum()
    #accuracy = ((correct / preds.shape[0])*100 + accuracy)/2
    accuracy = ((correct / batch_size)*100 + accuracy)/2

    #print(np.argmax(preds, axis=1), np.argmax(labels, axis=1), correct)
    #print( loss, preds, dout)

    ## printing
    t.set_description(f"Loss: {loss:.5f}; Acc: {accuracy :.5f}")
    accuracies.append(accuracy)
    losses.append(loss)
    
    net.optimize(lr=0.9, batch_size = batch_size)
    #net.save("CNN")

#plt.ylim(-0.01, 1.1)
plt.plot(losses)
plt.plot(accuracies)
plt.legend(["losses", "accuracies"])
#plt.show()
plt.savefig("stats.png")


