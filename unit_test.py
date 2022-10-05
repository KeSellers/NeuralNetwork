from DNN_utils.forward_utils import forward_activation_layer, forward_layer, forward
from DNN_utils.activation_utils import relu, relu_backward, sigmoid_backward,sigmoid
from DNN_utils.initialize_utils import initialize_params
from DNN_utils.cost_utils import compute_cost
from DNN_utils.backward_utils import backward_layer,backward_activation_layer,backward
import numpy as np
import unittest
from ressources.Testdata import activation_testdata, backward_testdata, forward_testdata, cost_testdata

epsilon = 1e-5
class TestActivations(unittest.TestCase):

    def test_relu(self):
        values,results = activation_testdata("relu")
        val1 , val2 , val3 = values
        res1 , res2 , res3 = results

        A1, cache1 = relu(val1) # positive scalar
        A2, cache2 = relu(val2) # negative scalar
        A3, cache3 = relu(val3) # matrix

        self.assertEqual(A1,res1,msg = "ReLu failed for Z = " + str(res1))
        self.assertEqual(A2,res2,msg = "ReLu failed for Z = " + str(res2))
        self.assertTrue((A3 == res3 ).all(), msg = "ReLu failed for Z = " + str(res2))
        self.assertEqual(val1,cache1,msg = "ReLu cache failed")
        self.assertEqual(val2,cache2,msg = "ReLu cache failed")
        self.assertTrue((val3 == cache3 ).all(), msg = "ReLu cache failed")

    def test_relu_backward(self):

        values,results = activation_testdata("relu_backward")
        val1 , val2 , val3 = values
        res1 , res2 , res3 = results

        dZ1 = relu_backward(val1 , val1)
        dZ2 = relu_backward(val2 , val2)
        dZ3 = relu_backward(val3 , val3)

        self.assertEqual(dZ1,res1,msg = "ReLu derivativ failed for Z = " + str(val1))
        self.assertEqual(dZ2,res2,msg = "ReLu derivativ failed for Z = " + str(val2))
        self.assertTrue((dZ3 == res3 ).all(), msg = "ReLu derivativ failed for Z = " + str(val2))

    def test_sigmoid(self):
        values,results = activation_testdata("sigmoid")
        val1 , val2 , val3 = values
        res1 , res2 , res3 = results

        A1, cache1 = sigmoid(val1)
        A2, cache2 = sigmoid(val2)
        A3, cache3 = sigmoid(val3)

        self.assertTrue(np.abs(A1 - res1) < epsilon ,msg = "Sigmoid failed for Z = " + str(val1))
        self.assertTrue(np.abs(A2 - res2) < epsilon, msg = "Sigmoid failed for Z = " + str(val2))
        self.assertTrue((np.abs(A3 - res3) < epsilon).all(),msg = "Sigmoid failed for Z = " + str(val3))
        self.assertEqual(val1 , cache1 , msg = "Sigmoid cache failed")
        self.assertEqual(val2 , cache2 , msg = "Sigmoid cache failed")
        self.assertTrue((val3 == cache3 ).all(), msg = "Sigmoid cache failed")

    def test_sigmoid_backward(self):

        values,results = activation_testdata("sigmoid_backward")
        val1 , val2 , val3 = values
        res1 , res2 , res3 = results

        A1 = sigmoid_backward(val1,val1)
        A2 = sigmoid_backward(val2,val2)
        A3 = sigmoid_backward(val3,val3)

        self.assertTrue(np.abs(A1 - res1) < epsilon ,msg = "Sigmoid derivative failed for dZ = " + str(val1))
        self.assertTrue(np.abs(A2 - res2) < epsilon, msg = "Sigmoid derivative failed for dZ = " + str(val2))
        self.assertTrue((np.abs(A3 - res3) < epsilon).all(),msg = "Sigmoid derivative failed for dZ = " + str(val3))
        
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
        parameters, result  = forward_testdata()
        A , W ,b = parameters

        Z, cache = forward_layer(A , W , b)
                            
        self.assertTrue((Z - result < epsilon).all(), "Error in Calcualtion of Z.")                                
        self.assertEqual(Z.shape,(W.shape[0],A.shape[1]),msg="Z has the wrong shape.")

class TestCost(unittest.TestCase):
    def test_compute_cost(self):
        A, Y, result = cost_testdata()
        cost = compute_cost(A,Y)
        self.assertTrue(cost - result <epsilon, msg = "Error in cost computation.")

class TestBackward(unittest.TestCase):
    def test_backward_layer(self):
        np.random.seed(1)
        dZ = np.random.rand(np.random.randint(1,3),np.random.randint(5))
        A_prev = np.random.rand(np.random.randint(1,3),dZ.shape[1])
        W = np.random.rand(dZ.shape[0],A_prev.shape[0])
        b = np.random.rand(dZ.shape[0],1)
        cache = (A_prev, W , b)
        dA, dW, db = backward_layer(dZ,cache)
        dA_res,dW_res,db_res = backward_testdata()


        self.assertTrue((dA - dA_res < epsilon).all(),"Calculation Error in backward_layer for dA.")
        self.assertTrue((dW - dW_res < epsilon).all(),"Calculation Error in backward_layer for dW.")
        self.assertTrue((db - db_res < epsilon).all(),"Calculation Error in backward_layer for db.")
    def test_backward_activation_layer(self):
        pass

    def test_backward_activation_layer(self):
        pass


unittest.main()

#========== NOT QUITE SURE IF THIS IS REALLY NECCESARY========
   # def test_forward_activation_layer(self):
       # np.random.seed(1) 
       # A_prev1 = np.random.randn(5,2) * 0.1
       # W1 = np.random.randn(3,A_prev1.shape[0]) * 0.1
       # b1 = np.zeros((W1.shape[0],1))
       # A1, cache1 = forward_activation_layer(A_prev1, W1, b1, activation="relu")
       # res1 = np.array([[0.02875668, 0.02067684],
       #         [0.     ,    0.0270083 ],
      #          [0.    ,     0.        ]])
       # A3, cache3 = forward_activation_layer(A_prev1, W1, b1, activation="sigmoid")
      #  res3 = np.array([[0.50718868, 0.50516903],
       #                 [0.49451112, 0.50675167],
       #                 [0.49888004, 0.49190786]])
       # np.random.seed(2)
       # A_prev2 = np.random.randn(5,2) * 0.1
       # W2 = np.random.randn(3,A_prev2.shape[0]) * 0.1
       # b2 = np.zeros((W2.shape[0],1))
       # A2, cache2 = forward_activation_layer(A_prev2, W2, b2, activation="relu")
       # res2 = np.array([[0.    ,     0.04595979],
        #        [0.      ,   0.        ],
        #        [0.00101209 ,0.01116153]])
       # A4, cache4 = forward_activation_layer(A_prev2, W2, b2, activation="sigmoid")
       # res4 = np.array([[0.48417177, 0.51148792],
       #                 [0.49449121, 0.49984055],
       #                 [0.50025302, 0.50279035]])
        #self.assertTrue((A1 - res1 < epsilon).all(), "Error in Calcualtion of A for Set 1.") 
        #self.assertTrue((A2 - res2 < epsilon).all(), "Error in Calcualtion of A for Set 2.") 
        #self.assertTrue((A3 - res3 < epsilon).all(), "Error in Calcualtion of A for Set 3.") 
        #self.assertTrue((A4 - res4 < epsilon).all(), "Error in Calcualtion of A for Set 4.") 
    #    pass
    #def test_foward(self):
        #np.random.seed(1)
        #layer_dims = [np.random.randint(1,10) for i in range(0, 10)]
        #parameters = initialize_params(layer_dims)
        #X = np.random.randn(layer_dims[0],np.random.randint(1,10))
        #AL, cache = forward(X,parameters)
        #res1 =[[0.5, 0.5, 0.5],
   #             [0.5, 0.5, 0.5],
    #            [0.5, 0.5, 0.5],
     #           [0.5, 0.5, 0.5],
     #           [0.5, 0.5, 0.5]]
    #    #self.assertEqual(AL.shape,(layer_dims[-1],X.shape[1]), msg = "Dimension-Error in calculation AL.")
        #self.assertTrue((AL - res1 < epsilon).all(), "Value-Error in Calcualtion of AL.")      
        #pass
