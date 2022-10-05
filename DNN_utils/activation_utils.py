import numpy as np
def relu(Z):
    A = np.maximum(0,Z)
    return A, Z
def sigmoid(Z):
    A =  1 / (1 + np.exp(-Z))
    return (A,Z)
def relu_backward(dA,cache):
    Z = cache
    dZ = np.array(dA, copy=True) 
    #dZ = dA * dZ < 0 * 1
    return dZ
def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ