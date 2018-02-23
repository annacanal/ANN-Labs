import numpy as np
import lab3_1
import matplotlib.pyplot as plt

# def patterns():
#     x1 = np.array([0, 0, 1, 0, 1, 0, 0, 1])
#     x2 = np.array([0, 0, 0, 0, 0, 1, 0, 0])
#     x3 = np.array([0, 1, 1, 0, 1, 0, 0, 1])
#     pattern = np.concatenate(([x1], [x2], [x3]))
#     return pattern

# def noisy_patterns():
#     x1d = np.array([1, 0, 1, 0, 1, 0, 0, 1])
#     x2d = np.array([1, 1, 0, 0, 0, 1, 0, 0,])
#     x3d = np.array([1, 1, 1, 0, 1, 1, 0, 1])
#     noisypattern = np.concatenate(([x1d], [x2d], [x3d]))
#     return noisypattern

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

def generate_weights(size):
    mu = np.zeros(size)
    cov = np.eye(size)
    weights = np.random.multivariate_normal(mu, cov, size)
    #Making it symmetric:
    # W = 0.5*(weights+weights.T)
    return weights

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
            if x[i][j] <  0:
                x[i][j] = 0
            else:
                x[i][j] = 1
    return x

def weight_matrix(nodes, patterns):
    W = np.zeros((nodes, nodes))
    for k in range(patterns.shape[0]):
        W += (1 / nodes) * (np.outer(np.transpose(patterns[k]), patterns[k]))
    return W

def calc_activations(W, input_pattern):
    old_output = input_pattern
    for i in range(5):
        new = np.sign(np.sum(W * old_output, axis=1))
        old_output = new
        e = energy(W, new)
        print(e)
    output = bipolar_binary(new.reshape((-1, 1)))
    output = output.flatten()
    return output

def seq_update(W, input_pattern):
    output= input_pattern
    #Pick a random number between 1 and 1024
    idx = np.random.randint(1, 1024)
    output[idx] = np.sum(W[idx]*input_pattern.T)
    output[output >= 0] = 1
    output[output < 0] = -1
    return output

def energy(weights, pattern):
    E = -1/2*(np.matmul(np.matmul(pattern, weights),np.transpose(pattern)))
    return E

def main():
    patterns_matrix = read_pictData()
    nodes = len(patterns_matrix[0])

    #train with p1, p2, p3
    train_patterns = np.concatenate(([patterns_matrix[0]], [patterns_matrix[1]], [patterns_matrix[2]]))
    # W = weight_matrix(nodes, train_patterns)
    W = generate_weights(nodes)
    print(W)

    #---------------------------------------outputs with batch
    # output1 = calc_activations(W, patterns_matrix[0])
    # output1 = calc_activations(W, patterns_matrix[1])
    # output1 = calc_activations(W, patterns_matrix[2])
    # print('and diatorted')
    # output1 = calc_activations(W, patterns_matrix[9])
    # output1 = calc_activations(W, patterns_matrix[10])
    #------------------------------------ outputs with sequential
    iterations = 7000
    E3 = []
    E4 = []
    for i in range(iterations):
        output3 = seq_update(W, patterns_matrix[9])
        # output4 = seq_update(W, patterns_matrix[10])
        E3.append(energy(W, output3))
        # E4.append(energy(W, output4))
    x = np.arange(0,iterations,1)
    plt.title("Energy values with random weight matrix (nonsymmetrical)")
    plt.plot(x,E3, label = 'pattern 9')
    # plt.plot(x,E4, label = 'pattern 10')
    plt.xlabel('Number of iterations')
    plt.ylabel('Energy vales')
    plt.legend()
    plt.show()

    
    

if __name__ == "__main__":
    main()

    # patterns_matrix = patterns()
    # print(patterns_matrix, 'pattern')

    # pattern = binary_bipolar(patterns_matrix)
    # noisy_pattern = noisy_patterns()
    # print(noisy_pattern, 'noise')
    # pattern_noise = binary_bipolar(noisy_pattern)
    # nodes = 8
    # weights = weight_matrix(nodes, pattern)

    # output1 = calc_activations(weights, pattern_noise[0])
    # output2 = calc_activations(weights, pattern_noise[1])
    # output3 = calc_activations(weights, pattern_noise[2])

    # print(output1, 'output_1')
    # print(output2, 'output_2')
    # print(output3, 'output_3')

