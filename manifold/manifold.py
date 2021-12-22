import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy


class KNN:
    def __init__(self, v_data):
        """
        @param v_data   list of feature vector(= 512 dim), list of ndarray
        """
        self.v_data = deepcopy(v_data)

    def GetDist(self, v_1, v_2):
        """
        @param v_1      feature vector(= 512 dim), ndarray
        @param v_2      feature vector(= 512 dim), ndarray
        @return         L2 norm of v_1, v_2, np.float64
        """
        return sum((v_1 - v_2) ** 2) ** 0.5

    def __call__(self, v_in, k = 10):
        """
        @param v_in     input feature vector(= 512 dim), ndarray
        @return         k-nearest neighbor feature vector, list of ndarray
        """
        # O(nlogn) implementation
        if k > len(self.v_data): return deepcopy(self.v_data)
        ret = sorted(self.v_data, key=lambda v: self.GetDist(v, v_in))
        return ret[:k]


class ConstrainedLeastSquareSolver:
    def __call__(self, v_in, v_k):
        """
        @param v_in     feature vector(= 512 dim), ndarray
        @param v_k      list of feature vector(= 512 dim), list of ndarray
        @return         v_proj, ndarray
        """
        k = len(v_k)
        C = np.array([[np.dot(v_in - v_k[i], v_in - v_k[j]) for j in range(k)] for i in range(k)])
        try:
            C_inv = np.linalg.inv(C)
            C_inv_sum = sum(C_inv[i][j] for i in range(k) for j in range(k))
            w = np.array([sum(C_inv[i][j] for j in range(k)) / C_inv_sum for i in range(k)])
            ret = sum(w[i] * v_k[i] for i in range(k))
            return ret
        except:
            return sum(v_k) / k
