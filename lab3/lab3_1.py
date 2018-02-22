import numpy as np


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

def weight_matrix(nodes, patterns):
    # # way 1
    # W1 = np.zeros((Nodes, Nodes))
    # for i in range(W1.shape[0]):
    #     for j in range(W1.shape[1]):
    #         for k in range(patterns.shape[0]):
    #             W1[i][j] += (1 / Nodes) * patterns[k][i] * patterns[k][j]

    W = np.zeros((nodes, nodes))
    for k in range(patterns.shape[0]):
        W += (1 / nodes) * (np.outer(np.transpose(patterns[k]), patterns[k]))
    return W


def calc_activations(nodes, W, input_pattern):
    return np.sum(W * input_pattern, axis=1)

def main():
    nodes = 8
    pattern = patterns()
    pattern_bin = binary_bipolar(pattern)
    W = weight_matrix(nodes, pattern)
    output = calc_activations(nodes, W, pattern[0])
    output[output>0] = 1
    output[output<=0] = 0
    print(output)
    # //print(bipolar_binary(output))



if __name__ == "__main__":
    main()