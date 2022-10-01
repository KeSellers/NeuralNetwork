import numpy as np
from .activation_utils import relu_backward,sigmoid_backward

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

    for l in reversed(range(L-1)):
        grads["dA"+ str(l)],grads["dW"+str(l+1)],grads["db"+str(l+1)] = backward_activation_layer(grads["dA" + str(l+1)], caches[l], "relu")

    return grads
