import numpy as np

from symbol import parameters


def initialize_params(self, X , layer_dims):
    #layer_dims [ input_layer, layer1 ,...., layern ]
    #parameters w(n,n-1) b(n-1,1)
    #He - initialization
    
    
    layers=len(layer_dims)
    parameters
    for l in range(layers):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) *  np.sqrt(2. / layer_dims[l-1])
        parameters["b" + str(l)] = np.zeros((layer_dims[l-1],layer_dims[l-1]))
    return parameters
def forward_layer(A , W, b):

    Z= np.dot( W , A ) +b
    cache = ( A, W, b)
    return Z, cache
def activation_layer(A_prev, W, b, activation="relu"):

    Z , linear_cache = forward_layer(A_prev,W,b)
    if activation=="relu":

        A , activation_cache = relu(Z)
    elif activation == "sigmoid":
        A , activation_cache = sigmoid(Z)
    else:
        raise ValueError('Activation has to be either "relu" or "sigmoid"')
    cache = (linear_cache, activation_cache)

    return A, cache
def relu(Z):
    A = np.max(0,Z)
    return A, Z
def sigmoid(Z):
    A =  1 / (1 + np.exp(-Z))
    return (A,Z)

def forward(X , parameters):

    L =  len(parameters) / 2
    A = X
    caches = []
    for l in range(L):
      A_prev = A
      A ,cache =  activation_layer(A_prev, parameters["W" + str(l)] , parameters["W" + str(l)] , "relu")
      caches.append(cache)
    AL ,cache = activation_layer(A, parameters["W" + str(L)] , parameters["W" + str(L)] , "sigmoid")
    caches.append(cache)

    return AL, caches




class NeuralNetwork():
    def __init__():
        pass
    def forward():
        pass
    def backward():
        pass
    def predict():
        pass

