import numpy as np
import lab3_1
import matplotlib.pyplot as plt


def read_pictData():
    number_patterns = 11  # 9 or 11 patterns of lenght = 1024
    patterns_matrix = np.zeros([number_patterns, 1024])

    with open("pict.dat", "r") as f:
        # Read the whole file at once
        patterns_line = f.read()
    patterns_line = patterns_line.split(",")
    for i in range(number_patterns):
        for j in range(1024):
            position = j + 1024 * i
            patterns_matrix[i][j] = patterns_line[position]
    return patterns_matrix


def random_pattern(row, column):
    y = np.random.uniform(-2, 2, (row, column))
    yy = np.sign(y)
    return yy

def make_bias(original_pattern, percentage):
    #as half of the randomly selected elements will already be ones, we need to select double of them
    percentage = percentage * 2
    biased_pattern = np.copy(original_pattern)
    # how many bits do we need to flip
    elements_num = biased_pattern.size * percentage / 100
    elements_num = int(elements_num)
    # select that many random indexes
    indexes = np.random.choice(original_pattern.size, elements_num, replace=False)
    # flip the bit
    for i, idx in enumerate(indexes):
        if biased_pattern[idx] == 0:
            biased_pattern[idx] = 1

    return biased_pattern


def pattern_transform(pattern):
    new_pattern = pattern.reshape(32, 32)
    return new_pattern


def binary_bipolar(x):
    for i in range(len(x)):
        for j in range(len(x[0])):
            if x[i][j] == 0:
                x[i][j] = -1
            else:
                x[i][j] = 1
    return x


def bipolar_binary(x):
    for i in range(len(x)):
        for j in range(len(x[0])):
            if x[i][j] < 0:
                x[i][j] = 0
            else:
                x[i][j] = 1
    return x


def weight_matrix(nodes, patterns, rho):
    W = np.zeros((nodes, nodes))
    for k in range(patterns.shape[0]):
        W += (np.outer(np.transpose(patterns[k]-rho), patterns[k]-rho))
    return W



def update(W, input_pattern, idx):
    # output = input_pattern
    output = np.sum(W * input_pattern.T)
    output[output >= 0] = 1
    output[output < 0] = -1
    new_output = output
    return new_output


def sync_update(W, input_pattern):
    old_output = input_pattern
    diffnum = 0
    loopnum = 0
    while diffnum < 5 and loopnum < 100000:
        # for i in range(5):
        new_output = np.sum(W * old_output, axis=1)
        new_output[new_output >= 0] = 1
        new_output[new_output < 0] = -1
        diff = np.sum(abs(old_output - new_output))
        if diff == 0:
            diffnum += 1
        else:
            diffnum = 0
        old_output = new_output
        loopnum += 1
    output = new_output
    iterations = 1  # loopnum - 10
    return output, iterations


def seq_update(W, input_pattern):
    output = input_pattern
    # Pick a random pattern element
    pat_length = input_pattern.size
    idx = np.random.randint(1, pat_length)
    output[idx] = np.sum(W[idx] * input_pattern.T)
    output[output >= 0] = 1
    output[output < 0] = -1
    return output

def seq_update_book(W, input_pattern, nodes, theta):
    output = np.zeros(nodes)
    order = np.arange(nodes)
    np.random.shuffle(order)
    for i in order:
        if np.sum(W[i,:]*input_pattern)>=0:
            output[i] = 1
        else:
            output[i] = 0
    return output

def energy(weights, pattern):
    E = -1 / 2 * (np.matmul(np.matmul(pattern, weights), np.transpose(pattern)))
    return E


def random_pattern(row, column):
    y = np.random.uniform(-2, 2, (row, column))
    yy = np.sign(y)
    return yy


def make_noise(original_pattern, percentage):
    noisy_pattern = np.copy(original_pattern)
    # how many bits do we need to flip
    elements_num = noisy_pattern.size * percentage / 100
    elements_num = int(elements_num)
    # select that many random indexes
    indexes = np.random.choice(original_pattern.size, elements_num, replace=False)
    # flip the bit
    for i, idx in enumerate(indexes):
        noisy_pattern[idx] = -1 if noisy_pattern[idx] == 1 else 1

    return noisy_pattern

def calc_activation(patterns):
    return (1/(np.shape(patterns)[0]*np.shape(patterns)[1])) * np.sum(patterns)


def main():
    original_patterns = np.zeros((300, 100))

    # add noise to the patterns that need to be recognised
    # percentage = 20  # noise percentage
    # noisy_patterns = np.zeros((np.shape(train_patterns)))
    # for i in range(noisy_patterns.shape[0]):
    #     noisy_patterns[i] = make_noise(train_patterns[i], percentage)

    # fig.suptitle("Synchronous update")
    theta_values = np.arange(0.1, 1, 0.1)
    for theta in [0.1,1,100]:#, 0.01, 0.1, 1, 10, 100, 1000, 10000]:#theta_values:
        all_capacity_percentage = np.zeros(original_patterns.shape[0])
        for times in [1, 2, 3, 4, 5]:

            # make sparse patterns
            bias_percentage = 5
            train_patterns = np.zeros((np.shape(original_patterns)))
            for i in range(original_patterns.shape[0]):
                train_patterns[i] = make_bias(original_patterns[i], bias_percentage)

            # initialize network
            nodes = len(original_patterns[0])

            capacity_percentage = []
            for i in range(train_patterns.shape[0]):
                patterns = train_patterns[0:i + 1]
                rho = calc_activation(patterns)
                W = weight_matrix(nodes, patterns, rho)
                saved = 0
                for j in range(i + 1):
                    output2 = seq_update_book(W, patterns[j], nodes, theta)
                    diff = np.sum(abs(output2 - patterns[j]))
                    if diff == 0:
                        saved = saved + 1
                capacity_percentage.append(saved * 100 / (i + 1))
                print(capacity_percentage)
            all_capacity_percentage = all_capacity_percentage + capacity_percentage
        plt.plot(np.arange(0, train_patterns.shape[0]), all_capacity_percentage/5, label = "bias: "+str(theta))



    plt.title("Capacity/patterns trained, with "+str(bias_percentage*2)+"% activity")
    plt.legend()
    plt.xlabel('Number of patterns')
    plt.ylabel('Capacity in percentage')

    plt.xlim((0, 50))
    plt.show()




if __name__ == "__main__":
    main()
