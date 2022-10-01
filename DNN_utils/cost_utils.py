import numpy as np

def compute_cost(A,Y):
    #cross entropy loss
    m = Y.shape[1]
    cost =-np.sum((np.multiply(Y,np.log(A)) , np.multiply((1-Y),np.log(1-A)))) / m 
    cost = np.squeeze(cost)
    return cost
