import numpy as np
from math import log2

# TODO add ADOM optimizer

class Layer:
  def __init__(self): #  just initials some varibles for  class
    self.params = {}
    self.cache = {}
    self.grads = {}

  def forward(self, input): raise NotImplementedError
  def backward(self, dout): raise NotImplementedError

  def im2col(self, input, HH, WW, stride=1, pad=0):
    N, C, H, W = input.shape
    out_h = int((H + 2*pad - HH) / stride) + 1
    out_w = int((W + 2*pad - WW) / stride) + 1
    input = np.pad(input, pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)
    col = np.zeros([N, out_h*out_w,C*HH*WW])
    for z in range(N):
      for i in range(out_h):
        for j in range(out_w):
          patch = input[z,...,i*stride:i*stride+HH,j*stride:j*stride+WW]
          col[z, i*out_w+j,:] = np.reshape(patch,-1)
    return col

  def col2im_back(self,input,out_h,out_w,stride,hh,ww,c):
    H = int((out_h  -1) * stride) + hh
    W = int((out_w  -1) * stride) + ww
    output = np.zeros([c,H,W])
    for i in range(out_h*out_w):
      row = input[i,:]
      h_start = int(i / out_w) * stride
      w_start = int(i % out_w) * stride
      output[:,h_start:h_start+hh,w_start:w_start+ww] += np.reshape(row,(c,hh,ww))
    return output


#   ****************************************************   #
class Convolution(Layer):
  def __init__(self, channels, num_filters, size_filters, stride=1, pad=0):
    super(Convolution, self).__init__()
    self.params["w"] = np.random.randn(num_filters, channels, size_filters, size_filters) * 0.01 
    self.params["b"] = np.zeros( num_filters )
    self.stride, self.pad = stride, pad

  def forward(self, input):
    self.cache["input"] = input 
    stride, pad = self.stride, self.pad
    N, C, H, W = self.cache["input"].shape 
    F, C, HH, WW = self.params["w"].shape  

    out_h = int((H + 2*pad - HH) / stride) + 1
    out_w = int((W + 2*pad - WW) / stride) + 1

    input  = self.im2col(input, HH, WW, stride, pad)  # input stack
    filter = self.params["w"].reshape(F, -1)          # filters stack
    bias   = self.params["b"].reshape(F, -1)          # bias stack

    # im2col convolution
    output = np.zeros((N, F, out_h*out_w))   # number of filters becomes new image channel
    for i in range(N):                       # iterating through image batchs
      image = input[i,:,:]
      output[i,:,:] = np.dot(filter, image.T) + bias
    output = output.reshape((N, F, out_h, out_w))
    return output

  def backward(self, dout):
    input = self.cache["input"] 
    stride, pad = self.stride, self.pad
    N, C, H, W = self.cache["input"].shape 
    F, C, HH, WW = self.params["w"].shape  

    out_h = int((H + 2*pad - HH) / stride) + 1
    out_w = int((W + 2*pad - WW) / stride) + 1

    output = np.zeros(input.shape)                    # number of filters becomes new image channel
    self.grads["w"] = np.zeros_like(self.params["w"]) # same size as params
    self.grads["b"] = np.zeros_like(self.params["b"]) # same size as params
     
    input  = self.im2col(input, HH, WW, stride, pad)  # input stack
    filter = self.params["w"].reshape(F, -1)          # filters stack

    # im2col convolution
    for i in range(N):                                # iterating through image batchs
      image = input[i,:,:]
      delta = dout[i,:,:,].reshape(F, -1)

      dx = np.dot(delta.T, filter)
      dx = self.col2im_back(dx,out_h,out_w,stride,HH,WW,C)
      output[i,:,:,:] = dx[:, pad:H+pad, pad:W+pad ]  # remove padding 

      self.grads["w"] += (np.dot(delta, image)).reshape(F,C,HH,WW)
      self.grads["b"] += np.sum(np.sum(dout.T), axis=(0))       # bias ????
    return output

#   ****************************************************   #
class MaxPooling(Layer):
  def __init__(self, size_filters):
    super(MaxPooling, self).__init__()
    self.stride, self.size = size_filters, size_filters

  def forward(self, input):         # TODO look at output looks wrong
    self.cache["input"] = input 
    stride, HH, WW = self.stride, self.size, self.size
    N, C, H, W = self.cache["input"].shape 

    out_h = int((H - HH) / stride) + 1
    out_w = int((W - WW) / stride) + 1
    input = self.im2col(input, HH, WW, stride, 0)  # input stack

    # im2col Poooling
    output = np.zeros((N, C, out_h*out_w))   # number of filters becomes new image channel
    for i in range(N):                       # iterating through image batchs
      image = input[i,:,:]
      output[i,:,:] = np.max(image, axis=1) 
    output = output.reshape((N, C, out_h, out_w))
    return output

  def backward(self, dout):
    input = self.cache["input"]
    stride, HH, WW = self.stride, self.size, self.size
    N, C, H, W = self.cache["input"].shape 

    out_h = int((H - HH) / stride) + 1
    out_w = int((W - WW) / stride) + 1

    output = np.zeros_like(input)
    for h in range(out_h):
      for w in range(out_w):
        patch = input[:, :, h*stride:h*stride+HH, w*stride:w*stride+WW]
        patch = patch.reshape(N, C, HH*WW)  # reshape into vector 

        maxvalue = dout[:,:,h,w]            # value of largests element in the patch vector
        position = np.argmax(patch, axis=2) # location of largests element in the patch vector
        grads = np.zeros_like(patch)        # creating a
        for n in range(N):
          for c in range(C):
            grads[n, c, position[n, c]] = maxvalue[n, c]
        output[:, :, h*stride:h*stride+HH, w*stride:w*stride+WW] += grads.reshape(N,C,HH,WW)
    return output


#   ****************************************************   #
class ReLU(Layer):
  def __init__(self):
    super(ReLU, self).__init__()
    pass

  def forward(self, input): 
    self.cache["input"] = input
    return np.maximum(input, 0)

  def backward(self, dout):
    grad = 1 * (self.cache["input"] > 0)
    return dout*grad

#   ****************************************************   #
class Linear(Layer):
  def __init__(self, input, output, flatten=False):
    super(Linear, self).__init__()
    self.params["w"] = np.random.randn(input, output)*0.01
    self.params["b"] = np.zeros(output)
    self.flatten = flatten

  def forward(self, input):
    if self.flatten:                                       # converts 2D to 1D
      self.cache["shape"] = input.shape
      input = input.reshape(input.shape[0], -1)

    self.cache["input"] = input
    return np.matmul(input, self.params["w"]) + self.params["b"]

  def backward(self, dout):
    tensor = self.cache["input"]
    self.grads["w"] = np.matmul(tensor.T, dout)
    self.grads["b"] = np.sum(dout, axis=0)
    dout = np.matmul(dout, self.params["w"].T)

    if self.flatten: dout = dout.reshape(self.cache["shape"])   # reshapes from 1D to 2D
    return dout

#   ****************************************************   #
class SoftmaxCrossEntropy:
  def __init__(self): pass

  # forward is stable softmax only
  def forward(self, X):
    e_x = np.exp(X - np.max(X)) # shift values
    return  e_x / np.sum(e_x, axis=1)[:, np.newaxis]

  # backward is softmax and cross entropy loss
  def backward(self, pred, y):

    # prevent log(0) error
    #min_nonzero = np.min(pred[np.nonzero(pred)])
    #pred[pred == 0] = min_nonzero

    loss = -np.sum(y * np.log(pred))   # Cross Entropy Loss
    dout = pred - y
    #print(loss, dout, pred, y)
    return dout, loss


#   ****************************************************   #
class Dropout(Layer):
  def __init__(self, p):
    super(Dropout, self).__init__()
    self.p = p
    self.mode = "train"

  def forward(self, input):
    p, mode = self.p, self.mode
    if mode == 'train':
      mask = np.random.choice([0, 1], size=input.shape, p=[p, 1 - p])
      output = input * mask / (1 - p)
      self.cache['mask'] = mask
    else:
      output = input
    return output

  def backward(self, dout):
    p, mask = self.p, self.cache['mask']
    dx = dout * mask / (1 - p)

    return dx


#   ****************************************************   #
class BatchNorm2d(Layer):
    def __init__(self, dim, epsilon=1e-5, momentum=0.9):
        super(BatchNorm2d, self).__init__()
        self.params['gamma'] = np.ones(dim)
        self.params['beta'] = np.zeros(dim)

        self.running_mean, self.running_var = np.zeros(dim), np.zeros(dim)
        self.epsilon, self.momentum = epsilon, momentum

        self.mode = "train"

    def forward(self, input):
        N, C, H, W = input.shape
        input = input.reshape(N * H * W, C)

        gamma, beta = self.params['gamma'], self.params['beta']
        running_mean, running_var = self.running_mean, self.running_var
        epsilon, momentum = self.epsilon, self.momentum

        output = 0
        if self.mode == 'train':
            mean, var = np.mean(input, axis=0), np.var(input, axis=0)
            norm = (input - mean) / np.sqrt(var + epsilon)
            output = gamma * norm + beta

            running_mean = momentum * running_mean + (1 - momentum) * mean
            running_var = momentum * running_var + (1 - momentum) * var

            self.running_mean, self.running_var = running_mean, running_var
            self.cache['input'], self.cache['norm'], self.cache['mean'], self.cache['var'] = input, norm, mean, var
        else:
            norm = (input - running_mean) / np.sqrt(running_var)
            output = gamma * norm + beta

        return output.reshape((N, C, H, W))

    def backward(self, dout):
        N1, C, H, W = dout.shape
        dout = dout.reshape(N1 * H * W, C)

        input, norm, mean, var = self.cache['input'], self.cache['norm'], self.cache['mean'], self.cache['var']
        gamma, beta = self.params['gamma'], self.params['beta']
        epsilon = self.epsilon
        N, _ = dout.shape

        self.grads['beta'] = np.sum(dout, axis=0)
        self.grads['gamma'] = np.sum(dout * norm, axis=0)

        dshift1 = 1 / (np.sqrt(var + epsilon)) * dout * gamma

        dshift2 = np.sum((input - mean) * dout * gamma, axis=0)
        dshift2 = (-1 / (var + epsilon)) * dshift2
        dshift2 = (0.5 / np.sqrt(var + epsilon)) * dshift2
        dshift2 = (2 * (input - mean) / N) * dshift2

        dshift = dshift1 + dshift2

        dx1 = dshift
        dx2 = -1 / N * np.sum(dshift, axis=0)
        dx = dx1 + dx2

        return dx.reshape((N1, C, H, W))


