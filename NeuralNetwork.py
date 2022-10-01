from symbol import parameters
import numpy as np




def initialize_params(layer_dims):
    #layer_dims [ input_layer, layer1 ,...., layern ]
    #parameters w(n,n-1) b(n-1,1)
    #He - initialization    
    layers=len(layer_dims)
    parameters={}
    for l in range(1,layers):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l] , layer_dims[l-1]) * np.sqrt(2. / layer_dims[l-1])
        parameters["b" + str(l)] = np.zeros((layer_dims[l],1))
        #print( str(parameters["b" + str(l)].shape) + "," + str(parameters["W" + str(l)].shape))
    return parameters
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
def relu(Z):
    A = np.maximum(0,Z)
    return A, Z
def sigmoid(Z):
    A =  1 / (1 + np.exp(-Z))
    return (A,Z)
def relu_backward(dA,cache):
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    return dZ
def sigmoid_backward(dA, cache):
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ
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

def compute_cost(A,Y):
    #cross entropy loss
    m = Y.shape[1]
    cost =-np.sum((np.multiply(Y,np.log(A)) , np.multiply((1-Y),np.log(1-A)))) / m 
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

def update(parameters, grads, lr):

    L = len(parameters) // 2
    for l in range(L):
        parameters["W"+str(l+1)] -= lr * grads["dW"+str(l+1)] 
        parameters["b"+str(l+1)] -= lr * grads["db"+str(l+1)] 

    return parameters
def model(X , Y, layer_dims, lr, n_iters, cache_cost=1000,print_cost=False):
    parameters = initialize_params(layer_dims)
    costs =[]
    for i in range(n_iters):
        AL,caches = forward(X,parameters)
        
        cost = compute_cost(AL,Y)
        grads = backward(AL , Y, caches)
        parameters = update(parameters, grads, lr)

        if print_cost and i % cache_cost == 0:
            costs.append(cost)
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))

    return parameters

class NeuralNetwork():
    def __init__():
        pass
    def forward():
        pass
    def backward():
        pass
    def predict():
        pass

def predict(X,parameters):
    AL,caches = forward(X,parameters)
    return AL