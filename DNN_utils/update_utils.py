def update(parameters, grads, lr):

    L = len(parameters) // 2
    for l in range(L):
        parameters["W"+str(l+1)] -= lr * grads["dW"+str(l+1)] 
        parameters["b"+str(l+1)] -= lr * grads["db"+str(l+1)] 

    return parameters