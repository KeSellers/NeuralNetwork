import numpy as np
from .activation_utils import sigmoid,relu

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
def forward(X , parameters , keep_prob):

    L =  len(parameters) // 2

    A = X
    caches = []
    Ds =[]
    D = np.random.rand(X.shape[0],X.shape[1]) < keep_prob 
    #A = A * D
    #A = A / keep_prob
    Ds.append(D)
    for l in range(1,L):
      A_prev = A

     #Dropout Reg

      A ,cache =  forward_activation_layer(A_prev, parameters["W" + str(l)] , parameters["b" + str(l)] , "relu")
      D = np.random.rand(A.shape[0],A.shape[1]) < keep_prob 
      A = A * D
      A = A / keep_prob
      Ds.append(D)


      cache = cache   
      caches.append(cache)

    AL ,cache = forward_activation_layer(A, parameters["W" + str(L)] , parameters["b" + str(L)] , "sigmoid")
    Ds.append(np.ones((A.shape[0],A.shape[1])))
    caches.append(cache)
    return AL, caches ,Ds