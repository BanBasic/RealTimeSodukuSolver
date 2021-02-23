import random
import time
import pickle
import numpy as np

from math import log2

import matplotlib.pyplot as plt
from tqdm import trange

import requests, gzip
from pathlib import Path

from src.MNISTLoader import *
from src.rose import *

class ANN: # cnn_digit_recognizer
  def __init__(self, size): 
    self.model = []
    for h, w in zip(size[:-1], size[1:]):
      self.model.append( Linear(h, w) )
      self.model.append( ReLU()  )

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
