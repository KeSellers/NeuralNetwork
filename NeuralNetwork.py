
import numpy as np
import matplotlib.pyplot as plt
from DNN_utils.forward_utils import forward
from DNN_utils.backward_utils import backward
from DNN_utils.initialize_utils import initialize_params, initialize_mini_batches
from DNN_utils.cost_utils import compute_cost,L2_Reg
from DNN_utils.update_utils import update
from Tests.test_model import test_nn
from DNN_utils.general_utils import save_model,load_model
from DNN_utils.activation_utils import relu_backward


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

    def __init__(self,layer_dims, lr = 0.01, n_iters = 1000, lambd=0.07, keep_prob = 1):

        self.layer_dims=layer_dims
        self.lr=lr
        self.n_iters=n_iters
        self.parameters = None
        self.costs = [] 
        self.lambd = lambd
        self.accuracy_train = None
        self.accuracy_dev = None
        self.keep_prob = keep_prob

    def train(self,X,Y,cache_cost=1000,print_cost=False):

        self.parameters = initialize_params(self.layer_dims)

        for i in range(self.n_iters):

            AL,caches,D = forward(X,self.parameters,self.keep_prob)
            cost = compute_cost(AL,Y)
            cost += L2_Reg(self.parameters,self.lambd)
            grads = backward(AL , Y, caches, self.lambd,self.keep_prob,D)
            self.parameters = update(self.parameters, grads, self.lr)

            if print_cost and i % cache_cost == 0:
                self.costs.append(cost)
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))

        Y_pred,_,_ = forward(X,self.parameters,self.keep_prob)
        self.accuracy_train = self.calc_accuracy(Y_pred,Y)
        print("Train-Accuracy: " + str(self.accuracy_train))
    def predict(self,X,Y):
        Y_pred, _, _ = forward(X,self.parameters,keep_prob=1) 
        self.accuracy_dev = self.calc_accuracy(Y_pred,Y)
        print("Dev-Accuracy: " + str(self.accuracy_dev))
    def calc_accuracy(self,Y_pred,Y):
        Y_pred = (Y_pred>=0.5) * 1.0
        return np.mean(Y_pred==Y)

    def plot_cost(self):
        step = self.n_iters//len(self.costs) 
        plt.plot(np.squeeze(self.costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per ' + str(step) +')')
        plt.title("Learning rate =" + str(self.lr))
        plt.show()
model = test_nn(NeuralNetwork)
model.plot_cost()
#save_model(model,"reg+dropout")
#nn = load_model("without_reg")
#print (nn.lambd)
#nn.plot_cost()

