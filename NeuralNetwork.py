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
def forward_activation_layer(A_prev, W, b, activation="relu"):

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
def relu_backward(Z,cache):
     Z = Z > 0 * 1.0 # all values greater than zero 
     return Z
def sigmoid_backward(Z, cache):
    Z = cache - np.square(cache)
    return Z
def forward(X , parameters):

    L =  len(parameters) // 2
    A = X
    caches = []
    for l in range(L):
      A_prev = A
      A ,cache =  forward_activation_layer(A_prev, parameters["W" + str(l)] , parameters["W" + str(l)] , "relu")
      caches.append(cache)
    AL ,cache = forward_activation_layer(A, parameters["W" + str(L)] , parameters["W" + str(L)] , "sigmoid")
    caches.append(cache)

    return AL, caches

def compute_cost(A,Y):
    #cross entropy loss
    m = Y.shape[1]
    cost = -np.sum(np.dot(Y,np.log(1-A)) + np.dot(1-Y,np.log(A))) / m
    cost = np.squeeze(cost)
    return cost
def backward_layer(dZ ,cache):

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ , A_prev.T)/m
    db = np.sum(dZ,keepdims=True,axis=1)/m
    dA_prev = np.dot(W.T,dZ)
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    return (dA_prev,dW,db)
    
    return dA_prev, dW, db
def backward_activation_layer(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation=="relu":

        dZ = relu_backward(dA,activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA,activation_cache)
    else:
        raise ValueError('Activation has to be either "relu" or "sigmoid"')
    dA_prev , dW, db = backward_layer(dZ,linear_cache)

    return dA_prev , dW, db
def backward (AL, Y, caches):
    grads = {}
    L = len(caches) # the number of layers -> 
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    grads["dA"+ str(L-1)],grads["dW"+str(L)],grads["db"+str(L)] = backward_activation_layer(dAL, caches[-1], "sigmoid")

    for l in reversed(range(L)):
        grads["dA"+ str(l)],grads["dW"+str(l+1)],grads["db"+str(l+1)] = backward_activation_layer(grads["dA" + str(l+1)], caches[l], "relu")

    return grads

def update (parameters, grads, lr):

    L = len(parameters) // 2
    for l in range(L):
        parameters["W"+str(l)] -= lr * grads["dW"+str(l)] 
        parameters["b"+str(l)] -= lr * grads["db"+str(l)] 

class NeuralNetwork():
    def __init__():
        pass
    def forward():
        pass
    def backward():
        pass
    def predict():
        pass

