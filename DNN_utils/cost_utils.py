import numpy as np

def compute_cost(A,Y):
    #cross entropy loss
    m = Y.shape[1]
    cost =-np.sum((np.multiply(Y,np.log(A)) , np.multiply((1-Y),np.log(1-A)))) / m 
    cost = np.squeeze(cost)
    return cost
def L2_Reg(parameters,lambd):
    m = parameters["W1"].shape[1]
    L = len(parameters) // 2
    for l in range(0,L):
        W_sum = 0
        W_sum += np.sum(np.square(parameters["W"+str(l+1)]))
    
    L2_regularization_cost = (lambd / (2 * m)) * W_sum
    return L2_regularization_cost