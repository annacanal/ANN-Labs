import numpy as np
import matplotlib.pyplot as plt

def generate_linearData():
    mean_x = [-3, -3]
    mean_y = [3, 3]
    cov_x = [[1, 0], [0, 1]]
    cov_y = [[1, 0], [0, 1]]
    N = 100

    x1, y1 = np.random.multivariate_normal(mean_x, cov_x, N).T
    x2, y2 = np.random.multivariate_normal(mean_y, cov_y, N).T

    classA = np.column_stack((x1, y1)).T
    classB = np.column_stack((x2, y2)).T

    # classA = np.row_stack((x1, y1))
    # classB = np.row_stack((x2, y2))

    X = np.append(classA[0], classB[0])
    Y = np.append(classA[1], classB[1])
    T = np.append(np.ones(len(classA[0])), -np.ones(len(classB[0])))
    bias = np.ones(2 * N)

    s = np.random.permutation(2 * N)
    patterns = np.concatenate(([X[s]], [Y[s]], [bias]), axis=0)
    target = T[s]

   # print(patterns)
    #print(target)

    # plt.plot(x1, y1,  'x')
    # plt.plot(x2, y2, 'o')
    # plt.axis('equal')
    # plt.show()
    return patterns, target

def generate_nonlinearData():
    mean_x = [3, 3]
    mean_y = [1, 1]
    cov_x = [[1, 0], [0, 1]]
    cov_y = [[1, 0], [0, 1]]
    N = 100

    x1, x2 = np.random.multivariate_normal(mean_x, cov_x, N).T
    y1, y2 = np.random.multivariate_normal(mean_y, cov_y, N).T

    classA = np.column_stack((x1, x2)).T
    classB = np.column_stack((y1, y2)).T

    X = np.append(classA[0], classB[0])
    Y = np.append(classA[1], classB[1])
    T = np.append(np.ones(len(classA[0])), -np.ones(len(classB[0])))
    bias = np.ones(2 * N)

    s = np.random.permutation(2 * N)
    patterns = np.concatenate(([X[s]], [Y[s]], [bias]), axis=0)
    target = T[s]

    # plt.plot(x1, y1,  'x')
    # plt.plot(x2, y2, 'o')
    # plt.axis('equal')
    # plt.show()
    return patterns, target
