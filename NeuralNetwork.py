
import numpy as np
import matplotlib.pyplot as plt
from DNN_utils.forward_utils import forward
from DNN_utils.backward_utils import backward
from DNN_utils.initialize_utils import initialize_params, initialize_mini_batches,initialize_adam
from DNN_utils.cost_utils import compute_cost,L2_Reg
from DNN_utils.update_utils import update,update_with_adam
from Tests.test_model import test_nn
from DNN_utils.general_utils import save_model,load_model

class NeuralNetwork():

    def __init__(self,layer_dims, lr = 0.01, n_epochs = 1000, lambd=0.07,
         keep_prob = 1,batch_size = 64, optimizer = "gd"):

        self.layer_dims=layer_dims
        self.lr=lr
        self.n_epochs=n_epochs
        self.parameters = None
        self.costs = [] 
        self.lambd = lambd
        self.accuracy_train = None
        self.accuracy_dev = None
        self.keep_prob = keep_prob
        self.batch_size = batch_size
        self.optimizer = optimizer

    def train(self,X,Y,cache_cost=1000,print_cost=False):
        
        #Initialize Parameters
        self.parameters = initialize_params(self.layer_dims)
        v,s = initialize_adam(self.parameters)
        m = X.shape[1]
   
        for i in range(self.n_epochs):
            
            #Initialize Epoch
            mini_batches = initialize_mini_batches(X, Y, self.batch_size)
            cost = 0

            for mini_batch in mini_batches:

                mini_batch_X,mini_batch_Y = mini_batch
                
                #Forward Mini-Batch
                AL,caches,D = forward(mini_batch_X,self.parameters,self.keep_prob)

                #Calculate Cost
                cost += compute_cost(AL,mini_batch_Y)
                cost += L2_Reg(self.parameters,self.lambd)

                #Backward Mini-Batch
                grads = backward(AL , mini_batch_Y, caches, self.lambd,self.keep_prob,D)

                #Update Mini-Batch
                if self.optimizer == "gd":
                    parameters = update(self.parameters, grads, self.lr)
                elif self.optimizer == "adam":
                    t = t + 1 # Adam counter
                    parameters, v, s = update_with_adam(parameters, grads, v, s,
                                                        t, self.lr, beta1 = 0.9, beta2 = 0.99,  epsilon = 10e-8)
                self.parameters = update(self.parameters, grads, self.lr)

            cost_avg = cost / m
            if print_cost and i % cache_cost == 0:
                self.costs.append(cost_avg)
                print("Cost after epoch {}: {}".format(i, np.squeeze(cost_avg)))

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
        step = self.n_epochs//len(self.costs) 
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

