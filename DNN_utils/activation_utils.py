import numpy as np
def relu(Z):
    A = np.maximum(0,Z)
    return A, Z
def sigmoid(Z):
    A =  1 / (1 + np.exp(-Z))
    return (A,Z)
def relu_backward(dA,cache):
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    #return np.greater(cache, 0).astype(int) why does this not work?
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0 # isnt this wrong?
    return dZ
def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ