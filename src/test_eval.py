import unittest
import eval
import numpy as np

class TestEvaluationMethods(unittest.TestCase):

    def test_empty_arrays(self):
        A = np.array([])
        B = np.array([])
        with self.assertRaises(ValueError):
            # 'testing empty numpy arrays'
            eval.mse(A, B) 
            eval.ssim(A, B)
            eval.kendall_tau(A, B)

    def test_one_empty_array(self):
        A = np.array([])
        B = np.array([[0], [1]])
        largeB = np.random.randint(0, 255, size=(200, 200))

        with self.assertRaises(ValueError):             
            # 'testing one empty numpy array'    
            eval.mse(A, B)
            eval.ssim(A, largeB)
            eval.kendall_tau(A, largeB)

    def test_same_array(self):
        A = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        B = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        largeA = np.random.randint(0, 255, size=(200, 200))

        self.assertEqual(eval.mse(A, B), 0, "testing mse on same array")
        # self.assertEqual(eval.ssim(largeA, largeA), 1, "testing ssim on same array")
        self.assertEqual(eval.ssim(largeA, largeA), 1, "testing kendall's tau on same array")

    def test_no_similarity(self):
        A = np.array([[4, 5, 6, 7], [0, 1, 2, 3]])
        B = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])

        arr = [[i for i in range(k-7, k)] for k in range(7, 50, 7)]
        ssimA = np.array(arr)
        ssimB = np.zeros((7,7))

        self.assertEqual(eval.mse(A, B)/16, 1, "no similarity")

    def test_other_cases(self):
        A = np.array([[4, 5, 6, 7], [0, 1, 2, 3]])
        B = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])

        arr = [[i for i in range(k-7, k)] for k in range(7, 50, 7)]
        ssimA = np.array(arr)
        ssimB = np.zeros((7,7))
        self.assertIsNotl(eval.ssim(ssimA, ssimB, 5), 1, "not similar")
        self.assertIsNot(eval.kendall_tau(A, B)[0], 1, "not similar")


if __name__ == '__main__':
    unittest.main()