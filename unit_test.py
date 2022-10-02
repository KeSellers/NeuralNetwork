from DNN_utils.forward_utils import forward_layer
from DNN_utils.activation_utils import relu, relu_backward
import numpy as np
import unittest
class TestForward(unittest.TestCase):
    def test_forward_layer(self):
        np.random.seed(1)
        epsilon = 10e-7
        A = np.random.randn(5,2) * 0.1
        W = np.random.randn(3,A.shape[0]) * 0.1
        b = np.zeros((W.shape[0],1))
        Z, cache = forward_layer(A , W , b)
        test_Z = np.array([[ 0.02875668 , 0.02067684],
                            [-0.02195641 , 0.0270083 ],
                            [-0.00447987,-0.03237137]])
                            
        self.assertTrue((Z - test_Z < epsilon).all(), "Error in Calcualtion of Z.")                                
        self.assertEqual(Z.shape,(W.shape[0],A.shape[1]),msg="Z has the wrong shape.")
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
        D3 = dZ3 > 0 * 1
        print (dZ1, dZ2, dZ3)
        self.assertEqual(dZ1,1,msg = "ReLu derivativ failed for Z = " + str(val1))
        self.assertEqual(dZ2,0,msg = "ReLu derivativ failed for Z = " + str(val2))
        self.assertTrue((dZ3 == D3 ).all(), msg = "ReLu derivativ failed for Z = " + str(val2))


unittest.main()