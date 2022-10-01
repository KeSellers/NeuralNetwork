import numpy as np
from activation_utils import sigmoid,relu

def forward_layer(A , W, b):
    #print(str(W.shape) + "," + str(A.shape) )
    Z= np.dot( W , A ) + b
    #print(Z.shape) 
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
def forward(X , parameters):

    L =  len(parameters) // 2

    A = X
    caches = []
    for l in range(1,L):
      A_prev = A
      A ,cache =  forward_activation_layer(A_prev, parameters["W" + str(l)] , parameters["b" + str(l)] , "relu")
      caches.append(cache)
    AL ,cache = forward_activation_layer(A, parameters["W" + str(L)] , parameters["b" + str(L)] , "sigmoid")
    caches.append(cache)

    return AL, caches