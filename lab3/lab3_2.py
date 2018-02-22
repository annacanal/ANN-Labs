import numpy as np
import lab3_1
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

def pattern_transform(pattern):
    new_pattern = pattern.reshape(32,32)
    return new_pattern


def weight_matrix(nodes, patterns):
    # # way 1
    # W1 = np.zeros((Nodes, Nodes))
    # for i in range(W1.shape[0]):
    #     for j in range(W1.shape[1]):
    #         for k in range(patterns.shape[0]):
    #             W1[i][j] += (1 / Nodes) * patterns[k][i] * patterns[k][j]

    W = np.zeros((nodes, nodes))
    for k in range(len(patterns)):
        W += (1 / nodes) * (np.outer(np.transpose(patterns[k]), patterns[k]))
    return W


def calc_activations(W, input_pattern):
    output = np.sum(W * input_pattern, axis=1)
    output[output >=0]=1
    output[output <0]=-1
    return output


def main():
    patterns_matrix = read_pictData()
    nodes=1024

    #train with p1, p2, p3
    train_patterns = [patterns_matrix[0], patterns_matrix[1], patterns_matrix[2]]
    W = weight_matrix(nodes, train_patterns)
    #plot p1, p2, p3
    p1_im = pattern_transform(patterns_matrix[0])
    plt.imshow(p1_im)
    plt.show()
    p2_im = pattern_transform(patterns_matrix[1])
    plt.imshow(p2_im)
    plt.show()
    p3_im = pattern_transform(patterns_matrix[2])
    plt.imshow(p3_im)
    plt.show()
    p9_im = pattern_transform(patterns_matrix[9])
    plt.imshow(p9_im)
    plt.show()
    p10_im = pattern_transform(patterns_matrix[10])
    plt.imshow(p10_im)
    plt.show()

    ### p11 and p22
    print("p11")
    print()
    print( patterns_matrix[9])
    print("p22")
    print(patterns_matrix[10])
    print()


############################# outputs
    output = calc_activations(W, patterns_matrix[9]) #check the p11 (which is the 10)
    output2 = calc_activations(W, patterns_matrix[10]) #check the p22 (which is the 11)

    output1_im = pattern_transform(output)
    plt.imshow(output1_im)
    plt.show()
    output2_im = pattern_transform(output2)
    plt.imshow(output2_im)
    plt.show()

if __name__ == "__main__":
    main()
