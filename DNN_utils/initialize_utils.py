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
