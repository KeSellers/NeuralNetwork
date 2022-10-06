from sklearn.datasets import make_circles
import numpy as np
import matplotlib.pyplot as plt
def load_dataset():
    np.random.seed(1)
    train_X, train_Y = make_circles(n_samples=3000, noise=.05)
    np.random.seed(2)
    test_X, test_Y = make_circles(n_samples=1000, noise=.05)
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
    layer_dims=[n_features, 3 , 3, 1]
    lr = 0.1
    n_iters = 10000
    nn = NeuralNetwork(layer_dims,lr,n_iters,lambd=0.,keep_prob=1)

    nn.train(train_X,train_Y,print_cost=True)

    nn.predict(test_X,test_Y)
    return nn

