import random
import time
import pickle
import numpy as np

class ANN:
  def __init__(self, size):
    self.grads, self.params, self.cache = {}, {}, {}
    self.num = len(size)
    self.size = size

    # weights and bias
    self.params["b"] = [ np.random.randn(col,   1)*0.01 for col in size[1:] ]
    self.params["w"] = [ np.random.randn(col, row)*0.01 for row, col in zip(size[:-1], size[1:]) ]

  def load(self, name):
    print('loading params')
    with open(name, 'rb') as handle:
        self.params = pickle.load(handle)
    pass

  def save(self, name):
    print('saving params')
    with open(name, 'wb') as handle:
        pickle.dump(self.params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pass

  def actFunc(self, input, deriv=False):

    ### tanh
    # if deriv: return np.tanh( 1-input**2 )
    #return np.tanh(inpu)

    ### sigmoiod
    if deriv: 
      a = 1 / (1 + np.exp(-input))
      return  a * (1 - a)
    return 1 / (1 + np.exp(-input))

    ### relu
    #if deriv: return 1. * (input > 0)
    #return np.maximum(input, 0)

    ### leaky relu
    #if deriv: return 1. * (input > 0.01)
    #return np.maximum(0.01*input, input)

  def forward(self, input):
    for weights, bias in zip(self.params["w"], self.params["b"]):
      input = np.dot(weights, input) + bias
    return input

  #def forward(self, X):
  #  e_x = np.exp(X - np.max(X))
  #  return  e_x / np.sum(e_x, axis=1)[:, np.newaxis]

  #  # backward is softmax and cross entropy loss
  #def backward(self, y_pred, y):

  #  # prevent log(0) error
  #  min_nonzero = np.min(y_pred[np.nonzero(y_pred)])
  #  y_pred[y_pred == 0] = min_nonzero

  #  loss = -np.sum(y * np.log(y_pred))   # Cross Entropy Loss
  #  dout = y_pred - y
  #  return dout, loss




  def backward(self, input, output):
    grads_b = [ np.zeros(b.shape) for b in self.params['b'] ]
    grads_w = [ np.zeros(w.shape) for w in self.params['w'] ]

    layer_vec = []                     # each layer's z vector
    layer_out = []                     # each layer's z vector after activation function
      
    # determinig the output of each layer
    layer_out.append( input )          # adding the input to update first layer
    for weights, bias in zip(self.params["w"], self.params["b"]):
      input = np.dot(weights, input) + bias
      layer_vec.append(input)

      input = self.actFunc(input) 
      layer_out.append(input)

    ouput = np.argmax(input) #* (1/256)
    error = int(ouput) - int(output)  # loss function

    delta = error * self.actFunc(layer_vec[-1], deriv=True)  
    grads_b[-1] = delta
    grads_w[-1] = np.dot(delta, layer_out[-2].T)

    # determing HIDDEN neurons weights and bias
    for l in range(2, self.num):
      delta = np.dot(self.params["w"][-l+1].T, delta) * self.actFunc( layer_vec[-l], True)  #  delta * dz/da * input 
      grads_b[-l] = delta
      grads_w[-l] = np.dot(delta, layer_out[-l-1].T)
    return error, grads_w, grads_b

  def train(self, epoch, batch_size, alpha, images, labels):
    lr, b1, b2, eps = 0.5, 0.9, 0.999, 1e-8
    grads = {}
    grads["m b"] = [np.zeros(b.shape) for b in self.params["b"]]
    grads["m w"] = [np.zeros(w.shape) for w in self.params["w"]]
    grads["rmsprop b"] = [np.zeros(b.shape) for b in self.params["b"]]
    grads["rmsprop w"] = [np.zeros(w.shape) for w in self.params["w"]]
    for iteration in range( epoch ):
      print("Iteration ", iteration)
      correct = 0 
      start = time.time()

      for i in range(0, len(images), batch_size):
        grads["b"] = [np.zeros(b.shape) for b in self.params["b"]]
        grads["w"] = [np.zeros(w.shape) for w in self.params["w"]]

        # collect grads
        for image, label in zip(images[i:i+batch_size], labels[i:i+batch_size]):
          image = image.reshape((784,1))
          error, grads_w, grads_b = self.backward(image, label)
          grads["b"] = [nb+dnb for nb, dnb in zip(grads["b"], grads_b)]
          grads["w"] = [nw+dnw for nw, dnw in zip(grads["w"], grads_w)]
          if (error == 0): correct +=1 

        #### SGD update rule
        #self.params["b"] = [b-(lr/batch_size)*nb for b, nb in zip(self.params["b"], grads["b"])]
        #self.params["w"] = [w-(lr/batch_size)*nw for w, nw in zip(self.params["w"], grads["w"])]

        ### Momentum + SDG update.
        grads["m b"] = [ b1*m + (1-b1)*g for m, g in zip(grads["m b"], grads["b"]) ]
        grads["m w"] = [ b1*m + (1-b1)*g for m, g in zip(grads["m w"], grads["w"]) ]
        self.params["b"] = [b-(lr/batch_size)*nb for b, nb in zip(self.params["b"], grads["m b"])]
        self.params["w"] = [w-(lr/batch_size)*nw for w, nw in zip(self.params["w"], grads["m w"])]


        #### RMS prop update rule
        #grads["rmsprop b"] = [b2*m + (1-b2)*np.power(g,2) for m, g in zip(grads["rmsprop b"], grads["b"]) ]
        #grads["rmsprop w"] = [b2*m + (1-b2)*np.power(g,2) for m, g in zip(grads["rmsprop w"], grads["w"]) ]
        #self.params["b"] = [b-(lr/batch_size)*(1/(np.sqrt(rms) +eps))*g for b, rms, g in zip(self.params["b"], grads["rmsprop b"], grads["b"])]
        #self.params["w"] = [w-(lr/batch_size)*(1/(np.sqrt(rms) +eps))*g for w, rms, g in zip(self.params["w"], grads["rmsprop w"], grads["w"])]

        #### ADAM update rule
        ## Momentum 
        #grads["m b"] = [ b1*m + (1-b1)*g for m, g in zip(grads["m b"], grads["b"]) ]
        #grads["m w"] = [ b1*m + (1-b1)*g for m, g in zip(grads["m w"], grads["w"]) ]

        ## RMS
        #grads["rmsprop b"] = [b2*m + (1-b2)*np.power(g,2) for m, g in zip(grads["rmsprop b"], grads["b"]) ]
        #grads["rmsprop w"] = [b2*m + (1-b2)*np.power(g,2) for m, g in zip(grads["rmsprop w"], grads["w"]) ]

        #self.params["b"] = [b-(lr/batch_size)*(1/(np.sqrt(rms) +eps))*g for b, rms, g in zip(self.params["b"], grads["rmsprop b"], grads["m b"])]
        #self.params["w"] = [w-(lr/batch_size)*(1/(np.sqrt(rms) +eps))*g for w, rms, g in zip(self.params["w"], grads["rmsprop w"], grads["m w"])]

        #print(np.mean(self.params["b"][1]), np.mean(self.params["w"][1]) )

        accuracy = 100*correct/len(images)
        print("Running accuracy: {:.3f} %".format( accuracy ), "| Error",error, end='\r')
        #if (iteration % 10 == 0 ): self.save( 'Params/params_'+str(iteration)+'.pickle')
      print("\nIteration took (s): {:.4f} ".format( time.time() - start), "\n" )
    pass

  #----------------------------------------------------------------------------------------------

def train_net():
  with open('MNISTData/MNISTData.pkl', 'rb') as f:
      data = pickle.load(f)

  shape = [784,30,10]          
  net = ANN( shape ) 

  SIZE = 30000             # number of data points to train and test with
  tSIZE = SIZE
  if (SIZE >= 10000): SIZE = 10000

  # max amount of training data avalibe is 6000
  training_images  = data['train_images'][:SIZE] #* (1/256) # normalising data
  training_labels  = data['train_labels'][:SIZE] #* (1/256) # normalising data

  # max amount of test data avalibe is 10000
  test_images  = data['test_images'][:tSIZE]     #* (1/256) # normalising data
  test_labels  = data['test_labels'][:tSIZE]     #* (1/256) # normalising data

  accuracy = 0
  print("\nTRAINING\n")
  #net.train( 1000, 10, 0.3, training_images , training_labels )

  print("\nTESTING\n")
  for label, image in zip(test_labels, test_images):
    out = net.forward(image.reshape((784,1)))

    if (np.argmax(out) == label) : accuracy +=1
    #if (label == np.argmax(out, axis=1)): accuracy += 1
    #print(label , np.argmax(out, axis=1) )
  print(" Accuracy: ", (accuracy/SIZE)*100, "%")
  print(" DONE ")
  pass

if __name__ == '__main__':
  #train_net()
  pass


