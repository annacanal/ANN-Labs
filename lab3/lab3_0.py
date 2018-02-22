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
            if x[i][j]== 0:
                x[i][j] = -1
            else:
                x[i][j] = 1
    return x

def bipolar_binary(x):
    # A value of zero will be translated to 1. 
    for i in range(len(x)):
        for j in range(len(x[0])):
            if x[i][j] < 0:
                x[i][j] = 0
            else:
                x[i][j] = 1
    return x

def weight_matrix(pattern):
    Nodes = len(pattern[0])
    W2 = np.zeros((Nodes,Nodes))
    for k in range(pattern.shape[0]):
        W2 += (1/Nodes) * ( np.outer(np.transpose(pattern[k]), pattern[k])  )
    return W2

def update_rule(weights, patterns): 
    new_pattern = np.zeros(len(weights))
    sum = 0
    for i in range(len(weights)):
        for j in range(len(weights)):
            sum += weights[i][j]*patterns[j]
        new_pattern[i] = np.sign(sum)
    return new_pattern

def main():
    pattern = patterns()
    pattern_bip = binary_bipolar(pattern)
    weights = weight_matrix(pattern_bip)
    new_pattern = update_rule(weights, pattern_bip[0])
    print(new_pattern)




if __name__ == "__main__":
    main()