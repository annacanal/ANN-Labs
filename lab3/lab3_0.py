import numpy as np
import matplotlib.pyplot as plt


def patterns():
    x1 = np.array([0, 0, 1, 0, 1, 0, 0, 1])
    x2 = np.array([0, 0, 0, 0, 0, 1, 0, 0])
    x3 = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    pattern = np.concatenate(([x1], [x2], [x3]))
    return pattern

def binary_bipolar(x):
    for i in range(len(x)):
        for j in range(len(x[0])):
            if x[i][j]==0:
                x[i][j] = -1
            else:
                x[i][j] = 1
    return x

def bipolar_binary(x):
    for i in range(len(x)):
        for j in range(len(x[0])):
            if x[i][j]==-1:
                x[i][j] = 0
            else:
                x[i][j] = 1
    return x

def weight_matrix(pattern):
    matrix = np.zeros((len(pattern),len(pattern[0])))
    for i in range(len(pattern)):
        for j in range(len(pattern[0])):
            matrix[i][j] = 
    return matrix

def main():
    pattern = patterns()
    pattern_bin = binary_bipolar(pattern)
    print(pattern_bin) 



if __name__ == "__main__":
    main()