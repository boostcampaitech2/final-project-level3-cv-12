from ManifoldProjection import *

sample = [
    np.array([1, 2, 3, 4]),
    np.array([2, 2, 3, 4]),
    np.array([1, 3, 3, 4]),
    np.array([0, 0, 3, 4]),
    np.array([0, 0, 0, 0]),
    np.array([10, 10, 10, 10])
]

v_in = np.array([1, 2, 3, 4])

knn = KNN(sample)
f = ConstrainedLeastSquareSolver()

for i in range(3, 6):
    v_proj = f(v_in, knn(v_in, i))
    print(f'current : {i}')
    print(f'{knn(v_in, i)=}')
    print(f'{v_proj=}')
    print()