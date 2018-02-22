import numpy as np
import matplotlib.pyplot as plt


def read_pictData():
    number_patterns = 11 #9 or 11 patterns of lenght = 1024
    patterns_matrix = np.zeros([number_patterns,1024])

    with open("pict.dat", "r") as f:
        # Read the whole file at once
        patterns_line = f.read()
    patterns_line = patterns_line.split(",")
    for i in range(number_patterns):
        for j in range(1024):
            position = j+1024*i
            patterns_matrix[i][j] = patterns_line[position]
    return patterns_matrix

def patterns_transform(patterns_matrix):
    new_patterns = []
    for i in range(patterns_matrix.shape[0]):
        new_pattern = patterns_matrix[i].reshape(32,32)
        new_patterns.append(new_pattern)
    return new_patterns


#-----------------above: same as for lab3_2------------

def weight_matrix(nodes, patterns):
    W = np.zeros((nodes, nodes))
    for k in range(patterns.shape[0]):
        W += (1 / nodes) * (np.outer(np.transpose(patterns[k]), patterns[k]))
    return W

def energy(weights, x):
    E = -1/2(np.matmul(np.matmul(x, weights),np.transpose(x)))


def main():
    patterns_matrix = read_pictData()
    patterns_transform(patterns_matrix)
    nodes = 1024
    weights = weight_matrix(nodes, patterns_matrix[0])
    # E = energy(weights, patterns_matrix[0])
    # print(E)
    y = np.matmul(np.transpose(patterns_matrix[0]), weights)
    yy  = np.matmul(y,patterns_matrix[0])    
    print(y.shape)
    print(yy)


if __name__ == "__main__":
    main()
