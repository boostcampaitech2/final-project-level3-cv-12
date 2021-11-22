import numpy as np
from copy import deepcopy

class KNN:
    def __init__(self, v_data):
        """
        @param v_data   list of feature vector(= 512 dim)
        """
        self.v_data = v_data

    def GetDist(self, v_1, v_2):
        """
        @param v_1      feature vector(= 512 dim)
        @param v_2      feature vector(= 512 dim)
        @return         L2 norm of v_1, v_2
        """
        return sum((v_1 - v_2) ** 2) ** 0.5

    def __call__(self, v_in, k = 10): # must be len(self.v_data) >= 10
        """
        @param v_in     input feature vector(= 512 dim)
        @return         k-nearest neighbor feature vector
        """
        # O(nlogn) implementation
        # self.v_data.sort(key=lambda v: self.GetDist(v, v_in))
        # return deepcopy(self.v_data[:k])

        # O(n) implementation
        ret = np.partition(self.v_data, k)
        return ret[:k]

class ConstrainedLeastSquareSolver:
    def __call__(self, v_in, v_k):
        """
        @param v_in     feature vector(= 512 dim)
        @param v_k      list of feature vector(= 512 dim)
        @return
        """
        k = len(v_k)
        C = np.zeros(shape=(k,k))
        for i in range(k):
            for j in range(k):
                C[i][j] = np.dot((v_in - v_k[i]), (v_in - v_k[j]))
        C_inv = np.linalg.inv(C)
        w = np.zeros(shape=(k))
        for i in range(k):
            a = sum(C_inv[i][j] for j in range(k))
            b = sum(C_inv[i][j] for i in range(k) for j in range(k))
            w[i] = a / b
        ret = np.zeros(shape=(512))
        for i in range(k):
            ret += w[i] * v_k[i]
        return ret