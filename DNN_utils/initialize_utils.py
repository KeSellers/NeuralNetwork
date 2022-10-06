import numpy as np
import math
def initialize_params(layer_dims):
    # layer_dims [ input_layer, layer1 ,...., layern ]
    # parameters w(n,n-1) b(n-1,1)
    # He - initialization    
    layers=len(layer_dims)
    parameters={}
    for l in range(1,layers):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l] , layer_dims[l-1]) * np.sqrt(2. / layer_dims[l-1])
        parameters["b" + str(l)] = np.zeros((layer_dims[l],1))

    return parameters

def initialize_mini_batches(X, Y, mini_batch_size = 64):

    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k + 1) * mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:

        mini_batch_X = shuffled_X[: ,-(m % mini_batch_size + 1) :-1]
        mini_batch_Y = shuffled_Y[: ,-(m % mini_batch_size + 1) :-1]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches