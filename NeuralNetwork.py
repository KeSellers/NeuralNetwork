
import numpy as np

from DNN_utils.forward_utils import forward
from DNN_utils.backward_utils import backward
from DNN_utils.initialize_utils import initialize_params
from DNN_utils.cost_utils import compute_cost
from DNN_utils.update_utils import update



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