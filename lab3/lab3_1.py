import numpy as np


def change_bit(pattern, bit_place):
    # change the bit
    the_bit = pattern[bit_place]
    if the_bit == -1:
        the_bit = 1
    else:
        the_bit = -1

    # replace the bit
    new_pattern = np.copy(pattern)
    new_pattern[bit_place] = the_bit

    return new_pattern


def make_all_noisy_patterns(patterns):
    #the noisy patterns will be:
    #for one (every possible) bit flipped: num_of_patterns x length_of_patterns
    noisy_patterns = np.zeros((np.shape(patterns)[0] * np.shape(patterns)[1], np.shape(patterns)[1]))
    for i in range(np.shape(patterns)[0]):
        the_original = patterns[i]
        for j in range(np.shape(patterns)[1]):
            the_new = change_bit(the_original, j)
            noisy_patterns[i*np.shape(patterns)[1] + j] = the_new

    #now change two bits
    for i in range(np.shape(patterns)[0]):
        the_original = patterns[i]
        for j in range(np.shape(patterns)[1]):
            for k in range(np.shape(patterns)[1]):
                if j == k: continue

                the_new = change_bit(the_original, j)
                the_new = change_bit(the_new, k)
                noisy_patterns = np.vstack((noisy_patterns, the_new))

    #now_change_three bits
    for i in range(np.shape(patterns)[0]):
        the_original = patterns[i]
        for j in range(np.shape(patterns)[1]):
            for k in range(np.shape(patterns)[1]):
                for l in range(np.shape(patterns)[1]):
                    if j == k or k == l or j == l: continue

                    the_new = change_bit(the_original, j)
                    the_new = change_bit(the_new, k)
                    the_new = change_bit(the_new, l)
                    noisy_patterns = np.vstack((noisy_patterns, the_new))


    #now_change_four bits
    for i in range(np.shape(patterns)[0]):
        the_original = patterns[i]
        for j in range(np.shape(patterns)[1]):
            for k in range(np.shape(patterns)[1]):
                for l in range(np.shape(patterns)[1]):
                    for m in range(np.shape(patterns)[1]):
                        if j == k or k == l or l == m or j == l or k == m or j == m: continue

                        the_new = change_bit(the_original, j)
                        the_new = change_bit(the_new, k)
                        the_new = change_bit(the_new, l)
                        the_new = change_bit(the_new, m)
                        noisy_patterns = np.vstack((noisy_patterns, the_new))

    return noisy_patterns



def noisy_patterns():
    x1d = np.array([1, 0, 1, 0, 1, 0, 0, 1])
    x2d = np.array([1, 1, 0, 0, 0, 1, 0, 0,])
    x3d = np.array([1, 1, 1, 0, 1, 1, 0, 1])
    noisypattern = np.concatenate(([x1d], [x2d], [x3d]))
    return noisypattern

def more_noisy_patterns():
    more_noisy1 = np.array([1, 1, 1, 1, 1, 1, 0, 0])  # 5 bits diff to learnt pattern 1
    more_noisy2 = np.array([0, 1, 1, 1, 0, 1, 1, 0])  # 5 bits diff to learnt pattern 2
    noisypattern = np.concatenate(([more_noisy1], [more_noisy2]))
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
    diffnum = 0
    loopnum = 0
    while diffnum < 10 and loopnum <1000000:

        new_output = np.sum(W * old_output, axis=1)
        new_output[new_output >= 0] = 1
        new_output[new_output < 0]= -1
        diff = np.sum(abs(old_output - new_output))
        if diff == 0:
            diffnum += 1
        else:
            diffnum = 0
        old_output = new_output
        loopnum += 1

    output = new_output
    output[output == -1] = 0
    return output

def main():
    nodes = 8
    pattern = patterns()
    pattern_bip = binary_bipolar(pattern)
    W = weight_matrix(nodes, pattern_bip)

    # check all given noise patterns
    # noisy_pattern = noisy_patterns()
    # noisy_bip = binary_bipolar(noisy_pattern)
    # output = calc_activations(W, noisy_bip[1])
    # print(output)

    #check all possible noisy patterns ubtil 4 bits
    # all_noisy = make_all_noisy_patterns(pattern_bip)
    # output = np.zeros(np.shape(all_noisy))
    # for i in range(all_noisy.shape[0]):
    #     output[i] = calc_activations(W, all_noisy[i])
    # attractors = np.unique(output, axis=0)
    # print(bipolar_binary(attractors))

    # check for even more noisy patterns
    more_noisy_pattern = more_noisy_patterns()
    noisy_bip = binary_bipolar(more_noisy_pattern)
    output = calc_activations(W, noisy_bip[0])
    print(output)
    output = calc_activations(W, noisy_bip[1])
    print(output)

    print()



if __name__ == "__main__":
    main()
