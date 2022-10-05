import numpy as np
def activation_testdata(test):
    val1 = 10
    val2 = -10
    if test == "relu":

        val3 = np.array([[ 0.02875668 , 0.02067684],
                            [-0.02195641 , 0.0270083 ],
                            [-0.00447987,-0.03237137]])
        res1 = 10
        res2 = 0
        res3 = np.array([[ 0.02875668 , 0.02067684],
                            [0 , 0.0270083 ],
                            [0,0]])
    elif test == "sigmoid":

        val3 = np.array([[ 0.02875668 , 0.02067684],
                            [-0.02195641 , 0.0270083 ],
                            [-0.00447987,-0.03237137]])
        res1 = 0.9999546021
        res2 = 0.0000453979
        res3 = np.array([[0.50718867, 0.50516903],
                            [0.49451112, 0.50675166],
                            [0.49888003, 0.49190786]])
    elif test == "relu_backward":

        val3 = np.array([[ 0.02875668 , 0.02067684],
                            [0. , 0. ],
                            [0.,0.]])
        res1 = 10
        res2 = 0
        res3 =np.array([[0.02875668, 0.02067684],
                [0.    ,     0.        ],
                [0.     ,    0.        ]])

    elif test == "sigmoid_backward":

        val3 = np.array([[ 0.02875668 , 0.02067684],
                            [-0.02195641 , 0.0270083 ],
                            [-0.00447987,-0.03237137]])
        res1 = 0.00045 
        res2 = -0.00045 
        res3 = np.array([[0.00718767,  0.00516865],
                            [-0.00548844 , 0.00675083],
                            [-0.00111996 ,-0.00809074]])
    values = (val1,val2,val3)
    results = (res1,res2,res3)

    return values, results

def backward_testdata():
    dA_res = np.array([[0.2994994,  0.08442023, 0.23964343],
                            [0.45204893, 0.04227877, 0.2467298 ]])
    db_res = np.array([[0.34092381],
                            [0.1417849 ]])
    dW_res = np.array([[0.24613628, 0.11992961],
                            [0.1141032,  0.0590028 ]])
    return dA_res,dW_res,db_res

def forward_testdata():
    np.random.seed(1)
    A = np.random.randn(5,2) * 0.1
    W = np.random.randn(3,A.shape[0]) * 0.1
    b = np.zeros((W.shape[0],1))
    result = np.array([[ 0.02875668 , 0.02067684],
                            [-0.02195641 , 0.0270083 ],
                            [-0.00447987,-0.03237137]])
    parameters = (A,W,b)
    return parameters,result

def cost_testdata():
    np.random.seed(4)
    A = np.random.rand(1,np.random.randint(1,15))
    Y = np.array([[np.random.randint(2) for _ in range(A.shape[1])]])
    result = 1.456309158894244
    return A, Y, result