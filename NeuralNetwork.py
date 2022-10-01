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

class NeuralNetwork():
    def __init__():
        pass
    def forward():
        pass
    def backward():
        pass
    def predict():
        pass

