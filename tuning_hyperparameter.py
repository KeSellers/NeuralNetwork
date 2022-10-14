from Tests.test_model import load_dataset
import numpy as np
import pandas as pd
def hyperparameter_tuning(NeuralNetwork):
    train_X, train_Y, test_X, test_Y = load_dataset()
    n_features,n_samples = train_X.shape
    # fix parameters
    batch_size = 64
    n_epochs = 10000
    n_layers = 4
    keep_prob = 1
    # parameter scales
    # lr = 0.0001 to 1
    # hidden layer1 10 to 70
    # hidden layer2 10 to 15
    # lambda 0.0001 to 0.1
    train_accuracy = []
    dev_accuracy = []
    lr=[]
    layer_dims = []
    lambd =[]
    header = {'lr':[],"layer_1":[],"layer_2":[],"lambda":[],"train_acc":[],"dev_acc":[]}
    data = pd.DataFrame(data = header)
    n_of_values = 5
    cnt = 0
    for i in range(n_of_values):
        
        r  = -4 * np.random.rand()
        lr.append( 10**r)
        for j in range(n_of_values):
            layer_dims.append([n_features,np.random.randint(10,70),np.random.randint(10,15),1])
            for k in range(n_of_values):
                cnt += 1
                print("Start {} of {} iterations.".format(cnt,n_of_values**3))
                l  = -4 * np.random.rand()
                lambd.append(10**l / 3)

                nn = NeuralNetwork(layer_dims[j],lr[i],n_epochs,lambd[k],keep_prob,batch_size=batch_size,optimizer="adam",decay_rate = 0.0001)
                train_accuracy = nn.train(train_X,train_Y,print_cost=False)

                dev_accuracy = nn.predict(test_X,test_Y)
                print("Training Accuracy: {} Dev Accuracy: {}".format(train_accuracy,dev_accuracy))
                layer_dim1 = layer_dims[j][1]
                layer_dim2 = layer_dims[j][2]
                d1 = pd.DataFrame({'lr':lr[i],"layer_1":[layer_dim1],"layer_2":[layer_dim2],"lambda":lambd[k],"train_acc":train_accuracy,"dev_acc":dev_accuracy})
                data = pd.concat([data,d1],ignore_index=True)
    data.to_excel("10000iter_more_data.xlsx",
             sheet_name='Sheet_name_1')