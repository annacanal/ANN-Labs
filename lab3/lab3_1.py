import numpy as np


def noisy_patterns():
    x1d = np.array([1, 0, 1, 0, 1, 0, 0, 1])
    x2d = np.array([1, 1, 0, 0, 0, 1, 0, 0,])
    x3d = np.array([1, 1, 1, 0, 1, 1, 0, 1])
    noisypattern = np.concatenate(([x1d], [x2d], [x3d]))
    return noisypattern

def patterns():
    x1 = np.array([0, 0, 1, 0, 1, 0, 0, 1])
    x2 = np.array([0, 0, 0, 0, 0, 1, 0, 0])
    x3 = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    pattern = np.concatenate(([x1], [x2], [x3]))
    return pattern

def binary_bipolar(x):
    y = np.zeros(np.shape(x))
    for i in range(len(x)):
        for j in range(len(x[0])):
            if x[i][j]==0:
                y[i][j] = -1
            else:
                y[i][j] = 1
    return y

def bipolar_binary(x):
    y = np.zeros(np.shape(x))
    for i in range(len(x)):
        for j in range(len(x[0])):
            if x[i][j] <=  0:
                y[i][j] = 0
            else:
                y[i][j] = 1
    return y

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


def calc_activations(W, input_pattern):
    old_output = input_pattern
    for i in range(100):

        new_output = np.sum(W * old_output, axis=1)
        # new_output[new_output >= 0] = 1
        # new_output[new_output < 0]= -1
        old_output = new_output

    new_output[new_output >= 0] = 1
    new_output[new_output < 0]= -1
    # output = bipolar_binary(new_output.reshape((-1, 1)))
    # output = output.flatten()
    output = new_output
    output[output == -1] = 0
    return output

def main():
    nodes = 8
    pattern = patterns()
    pattern_bip = binary_bipolar(pattern)
    W = weight_matrix(nodes, pattern_bip)

    noisy_pattern = noisy_patterns()
    noisy_bip = binary_bipolar(noisy_pattern)
    output = calc_activations(W, pattern_bip[0])
    # print("output:", output, "pattern:", pattern[0], "input:", pattern[0])
    # output = calc_activations(W, pattern_bip[1])
    # print("output:", output, "pattern:", pattern[1], "input:", pattern[1])
    # output = calc_activations(W, pattern_bip[2])
    # print("output:", output, "pattern:", pattern[2], "input:", pattern[2])
    #
    # output = calc_activations(W, noisy_bip[0])
    # print("output:", output, "pattern:", pattern[0], "input:", noisy_pattern[0])
    output = calc_activations(W, noisy_bip[1])
    print(output)
    # print("output:",output, "pattern:", pattern[1], "input:", noisy_pattern[1])
    # output = calc_activations(W, noisy_bip[2])
    # print("output:", output, "pattern:", pattern[2], "input:", noisy_pattern[2])



if __name__ == "__main__":
    main()
