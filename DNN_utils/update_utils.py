import numpy as np
def update(parameters, grads, lr):

    L = len(parameters) // 2
    for l in range(L):
        parameters["W"+str(l+1)] -= lr * grads["dW"+str(l+1)] 
        parameters["b"+str(l+1)] -= lr * grads["db"+str(l+1)] 

    return parameters

def update_with_adam(parameters, grads, v, s,t, learning_rate, beta1 = 0.9,
                                                            beta2 = 0.99,  epsilon = 10e-8):
    
    L = len(parameters) // 2                 
    v_corrected = {}                        
    s_corrected = {} 

    for l in range(L):

        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads["db" + str(l + 1)]

        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".

        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1-beta1**t) 
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1-beta1**t) 


        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".

        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.square(grads["dW" + str(l + 1)])
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.square(grads["db" + str(l + 1)])


        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".

        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1-beta2**t) 
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1-beta2**t)

        parameters["W" + str(l + 1)] -= learning_rate * (v_corrected["dW" + str(l + 1)] / (np.sqrt(s_corrected["dW" + str(l + 1)]) + epsilon))
        parameters["b" + str(l + 1)] -= learning_rate * (v_corrected["db" + str(l + 1)] / (np.sqrt(s_corrected["db" + str(l + 1)]) + epsilon))
        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters"
    return parameters, v, s