import numpy as np
import lab3_1
import matplotlib.pyplot as plt

def patterns():
    x1 = np.array([0, 0, 1, 0, 1, 0, 0, 1])
    x2 = np.array([0, 0, 0, 0, 0, 1, 0, 0])
    x3 = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    pattern = np.concatenate(([x1], [x2], [x3]))
    return pattern

def noisy_patterns():
    x1d = np.array([1, 0, 1, 0, 1, 0, 0, 1])
    x2d = np.array([1, 1, 0, 0, 0, 1, 0, 0,])
    x3d = np.array([1, 1, 1, 0, 1, 1, 0, 1])
    noisypattern = np.concatenate(([x1d], [x2d], [x3d]))
    return noisypattern

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
            if x[i][j] <=  0:
                x[i][j] = 0
            else:
                x[i][j] = 1
    return x

def weight_matrix(nodes, patterns):
    W = np.zeros((nodes, nodes))
    for k in range(patterns.shape[0]):
        W += (1 / nodes) * (np.outer(np.transpose(patterns[k]), patterns[k]))
    return W

# def calc_activations(W, input_pattern):
#     output = []
#     for i in range(len(input_pattern)):
#         o = np.sum(W * input_pattern[i], axis=1)
#         output.append(o)
#     output = np.concatenate((output[0], output[1], output[2]))
#     return output

def calc_activations(W, input_pattern):
    output = np.sum(W * input_pattern, axis=1)
    output = bipolar_binary(output.reshape((-1, 1)))
    output = output.flatten()
    return output


def energy(weights, pattern):
    E = -1/2*(np.matmul(np.matmul(pattern, weights),np.transpose(pattern)))
    return E

def main():
    patterns_matrix = patterns()
    pattern = binary_bipolar(patterns_matrix)

    noisy_pattern = noisy_patterns()
    pattern_noise = binary_bipolar(noisy_pattern)
    
    nodes = 8
    weights = weight_matrix(nodes, pattern)

    E = energy(weights, pattern_noise[0])
    print(E)


if __name__ == "__main__":
    main()


# def read_pictData():
#     number_patterns = 11 #9 or 11 patterns of lenght = 1024
#     patterns_matrix = np.zeros([number_patterns,1024])

#     with open("pict.dat", "r") as f:
#         # Read the whole file at once
#         patterns_line = f.read()
#     patterns_line = patterns_line.split(",")
#     for i in range(number_patterns):
#         for j in range(1024):
#             position = j+1024*i
#             patterns_matrix[i][j] = patterns_line[position]
#     return patterns_matrix

# def pattern_transform(pattern):
#     new_pattern = pattern.reshape(32,32)
#     return new_pattern
