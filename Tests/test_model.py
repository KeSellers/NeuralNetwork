from sklearn.datasets import make_circles
import numpy as np
import matplotlib.pyplot as plt
from DNN_utils.forward_utils import forward
def load_dataset():
    np.random.seed(1)
    train_X, train_Y = make_circles(n_samples=600, noise=.05)
    np.random.seed(2)
    test_X, test_Y = make_circles(n_samples=200, noise=.05)
    # Visualize the data
    #plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    return train_X, train_Y, test_X, test_Y



def test_nn (NeuralNetwork):
    train_X, train_Y, test_X, test_Y = load_dataset()
    n_features,n_samples = train_X.shape
    layer_dims=[n_features, 30 , 15, 5,1]
    lr = 0.007
    n_epochs = 10000
    batch_size = 64
    nn = NeuralNetwork(layer_dims,lr,n_epochs,lambd=0.04,keep_prob=0.99,batch_size=batch_size,optimizer="adam",decay_rate = 0.0001)

    nn.train(train_X,train_Y,print_cost=True)

    nn.predict(test_X,test_Y)
    plot_decision_boundary(lambda x: predict_dec(nn.parameters, x.T),train_X,train_Y)
    plot_decision_boundary(lambda x: predict_dec(nn.parameters, x.T),test_X,test_Y)
    return nn

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()
def predict_dec(parameters, X):

    # Predict using forward propagation and a classification threshold of 0.5
    a3, cache, _ = forward(X, parameters,keep_prob=1)
    predictions = (a3 > 0.5)
    return predictions