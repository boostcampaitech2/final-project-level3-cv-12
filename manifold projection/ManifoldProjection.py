from copy import deepcopy

class KNN: # O(nlogn) implementation, should be revised to O(n) using nth_element
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

    def __call__(self, v_in, k = 10):
        """
        @param v_in     input feature vector(= 512 dim)
        @return         k-nearest neighbor feature vector
        """
        self.v_data.sort(key=lambda v: self.GetDist(v, v_in))
        return deepcopy(self.v_data[:k])

class ConstrainedLeastSquareSolver:
    def __call__(self, v_in, v_k):
        """
        @param v_in     feature vector(= 512 dim)
        @param v_k      list of feature vector(= 512 dim)
        @return
        """
        k = len(v_k)
        # WIP
        pass