from DNN_utils.forward_utils import forward_activation_layer, forward_layer, forward
from DNN_utils.activation_utils import relu, relu_backward, sigmoid_backward,sigmoid
from DNN_utils.initialize_utils import initialize_params
from DNN_utils.cost_utils import compute_cost
import numpy as np
import unittest

epsilon = 1e-5
class TestActivations(unittest.TestCase):
    def test_relu(self):
        val1 = 10
        val2 = -10
        val3 = np.array([[ 0.02875668 , 0.02067684],
                            [-0.02195641 , 0.0270083 ],
                            [-0.00447987,-0.03237137]])
        A1, cache1 = relu(val1)
        A2, cache2 = relu(val2)
        A3, cache3 = relu(val3)
        D3 = A3 > 0 * 1
        D3 = A3 * D3
        self.assertEqual(A1,val1,msg = "ReLu failed for Z = " + str(val1))
        self.assertEqual(A2,0,msg = "ReLu failed for Z = " + str(val2))
        self.assertTrue((A3 == D3 ).all(), msg = "ReLu failed for Z = " + str(val2))
        self.assertEqual(val1,cache1,msg = "ReLu cache failed")
        self.assertEqual(val2,cache2,msg = "ReLu cache failed")
        self.assertTrue((val3 == cache3 ).all(), msg = "ReLu cache failed")
    def test_relu_backward(self):
        val1 = 10
        val2 = 0
        val3 = np.array([[ 0.02875668 , 0.02067684],
                            [0. , 0. ],
                            [0.,0.]])

        dZ1 = relu_backward(val1 , val1)
        dZ2 = relu_backward(val2 , val2)
        dZ3 = relu_backward(val3 , val3)
        D3 =dZ3 * (dZ3 > 0 * 1)
        #print (dZ1, dZ2, dZ3)
        self.assertEqual(dZ1,10,msg = "ReLu derivativ failed for Z = " + str(val1))
        self.assertEqual(dZ2,0,msg = "ReLu derivativ failed for Z = " + str(val2))
        self.assertTrue((dZ3 == D3 ).all(), msg = "ReLu derivativ failed for Z = " + str(val2))   
    def test_sigmoid(self):
        val1 = 10
        val2 = -10
        val3 = np.array([[ 0.02875668 , 0.02067684],
                            [-0.02195641 , 0.0270083 ],
                            [-0.00447987,-0.03237137]])
        val4 = 10000
        A1, cache1 = sigmoid(val1)
        A2, cache2 = sigmoid(val2)
        A3, cache3 = sigmoid(val3)
        A4, cache4 = sigmoid(val4)
        res1 = 0.9999546021
        res2 = 0.0000453979
        res3 = np.array([[0.50718867, 0.50516903],
                            [0.49451112, 0.50675166],
                            [0.49888003, 0.49190786]])
        res4 = 1.
        self.assertTrue(np.abs(A1 - res1) < epsilon ,msg = "Sigmoid failed for Z = " + str(val1))
        self.assertTrue(np.abs(A2 - res2) < epsilon, msg = "Sigmoid failed for Z = " + str(val2))
        self.assertTrue((np.abs(A3 - res3) < epsilon).all(),msg = "Sigmoid failed for Z = " + str(val3))
        self.assertTrue(np.abs(A4 - res4) < epsilon,msg = "Sigmoid failed for Z = " + str(val4))
        self.assertEqual(val1 , cache1 , msg = "Sigmoid cache failed")
        self.assertEqual(val2 , cache2 , msg = "Sigmoid cache failed")
        self.assertTrue((val3 == cache3 ).all(), msg = "Sigmoid cache failed")
        self.assertEqual(val4 , cache4,msg = "Sigmoid cache failed")
    def test_sigmoid_backward(self):
        val1 = 10
        val2 = -10
        val3 = np.array([[ 0.02875668 , 0.02067684],
                            [-0.02195641 , 0.0270083 ],
                            [-0.00447987,-0.03237137]])
        val4 = 10000
        A1 = sigmoid_backward(val1,val1)
        A2 = sigmoid_backward(val2,val2)
        A3 = sigmoid_backward(val3,val3)
        A4 = sigmoid_backward(val4,val4)
        res1 = 0.000045 * val1
        res2 = 0.000045 * val2
        res3 = np.array([[0.249948, 0.249973],
                            [0.24997, 0.249954],
                            [0.249999, 0.249935]])
        res3 = np.multiply(val3,res3)
        res4 = 0 * val4

        self.assertTrue(np.abs(A1 - res1) < epsilon ,msg = "Sigmoid derivative failed for dZ = " + str(val1))
        self.assertTrue(np.abs(A2 - res2) < epsilon, msg = "Sigmoid derivative failed for dZ = " + str(val2))
        self.assertTrue((np.abs(A3 - res3) < epsilon).all(),msg = "Sigmoid derivative failed for dZ = " + str(val3))
        self.assertTrue(np.abs(A4 - res4) < epsilon,msg = "Sigmoid derivative failed for dZ = " + str(val4))

class TestInitialisation(unittest.TestCase):
    def test_initialisation_He(self):
        np.random.seed(1)
        layer_dims = [np.random.randint(1,10) for i in range(0, 10)]
        parameters = initialize_params(layer_dims)
        L = len(layer_dims)
        for l in range(1,L):
            self.assertEqual(parameters["W" + str (l)].shape,(layer_dims[l],layer_dims[l-1]),msg="W" + str(l)+" has the wrong shape.")
            self.assertEqual(parameters["b" + str (l)].shape,(layer_dims[l],1),msg="b" + str(l)+" has the wrong shape.")
            
class TestForward(unittest.TestCase):
    def test_forward_layer(self):
        np.random.seed(1)
        A = np.random.randn(5,2) * 0.1
        W = np.random.randn(3,A.shape[0]) * 0.1
        b = np.zeros((W.shape[0],1))
        Z, cache = forward_layer(A , W , b)
        res1 = np.array([[ 0.02875668 , 0.02067684],
                            [-0.02195641 , 0.0270083 ],
                            [-0.00447987,-0.03237137]])
                            
        self.assertTrue((Z - res1 < epsilon).all(), "Error in Calcualtion of Z.")                                
        self.assertEqual(Z.shape,(W.shape[0],A.shape[1]),msg="Z has the wrong shape.")
    def test_forward_activation_layer(self):
        np.random.seed(1) 
        A_prev1 = np.random.randn(5,2) * 0.1
        W1 = np.random.randn(3,A_prev1.shape[0]) * 0.1
        b1 = np.zeros((W1.shape[0],1))
        A1, cache1 = forward_activation_layer(A_prev1, W1, b1, activation="relu")
        res1 = np.array([[0.02875668, 0.02067684],
                [0.     ,    0.0270083 ],
                [0.    ,     0.        ]])
        A3, cache3 = forward_activation_layer(A_prev1, W1, b1, activation="sigmoid")
        res3 = np.array([[0.50718868, 0.50516903],
                        [0.49451112, 0.50675167],
                        [0.49888004, 0.49190786]])
        np.random.seed(2)
        A_prev2 = np.random.randn(5,2) * 0.1
        W2 = np.random.randn(3,A_prev2.shape[0]) * 0.1
        b2 = np.zeros((W2.shape[0],1))
        A2, cache2 = forward_activation_layer(A_prev2, W2, b2, activation="relu")
        res2 = np.array([[0.    ,     0.04595979],
                [0.      ,   0.        ],
                [0.00101209 ,0.01116153]])
        A4, cache4 = forward_activation_layer(A_prev2, W2, b2, activation="sigmoid")
        res4 = np.array([[0.48417177, 0.51148792],
                        [0.49449121, 0.49984055],
                        [0.50025302, 0.50279035]])
        self.assertTrue((A1 - res1 < epsilon).all(), "Error in Calcualtion of A for Set 1.") 
        self.assertTrue((A2 - res2 < epsilon).all(), "Error in Calcualtion of A for Set 2.") 
        self.assertTrue((A3 - res3 < epsilon).all(), "Error in Calcualtion of A for Set 3.") 
        self.assertTrue((A4 - res4 < epsilon).all(), "Error in Calcualtion of A for Set 4.") 
    def test_foward(self):
        np.random.seed(1)
        layer_dims = [np.random.randint(1,10) for i in range(0, 10)]
        parameters = initialize_params(layer_dims)
        X = np.random.randn(layer_dims[0],np.random.randint(1,10))
        AL, cache = forward(X,parameters)
        res1 =[[0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5]]
        self.assertEqual(AL.shape,(layer_dims[-1],X.shape[1]), msg = "Dimension-Error in calculation AL.")
        self.assertTrue((AL - res1 < epsilon).all(), "Value-Error in Calcualtion of AL.")      

class TestCost(unittest.TestCase):
    def test_compute_cost(self):
        np.random.seed(4)
        A = np.random.rand(1,np.random.randint(1,15))
        Y = np.array([[np.random.randint(2) for _ in range(A.shape[1])]])
        res1 = 1.456309158894244
        cost = compute_cost(A,Y)
        self.assertTrue(cost - res1 <epsilon, msg = "Error in cost computation.")

unittest.main()


