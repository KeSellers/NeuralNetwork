from sklearn.datasets import make_circles
from NeuralNetwork import model , predict
import numpy as np
import matplotlib.pyplot as plt
def load_dataset():
    np.random.seed(1)
    train_X, train_Y = make_circles(n_samples=300, noise=.05)
    np.random.seed(2)
    test_X, test_Y = make_circles(n_samples=100, noise=.05)
    # Visualize the data
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    return train_X, train_Y, test_X, test_Y

train_X, train_Y, test_X, test_Y = load_dataset()
n_features,n_samples = train_X.shape
layer_dims=[n_features, 3 , 3, 1]
lr = 0.1
n_iters = 10000
parameters = model(train_X, train_Y, layer_dims,lr,n_iters,print_cost=True)
y_pred = predict(test_X , parameters)
p = np.zeros((1,y_pred.shape[1]))
for i in range(0, y_pred.shape[1]):
    if y_pred[0,i] > 0.5:
        p[0,i] = 1
    else:
        p[0,i] = 0
print("Accuracy: "  + str(np.mean((p[0,:] == test_Y[0,:]))))