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
        if biased_pattern[idx] == -1:
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


def weight_matrix(nodes, patterns):
    W = np.zeros((nodes, nodes))
    for k in range(patterns.shape[0]):
        W += (1 / nodes) * (np.outer(np.transpose(patterns[k]), patterns[k]))
    return W


def weight_matrix_zeroDiag(W):
    Wnew = np.copy(W)
    for i in range(W.shape[0]):
        Wnew[i][i] = 0

    return Wnew


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
    # print("iterations:")
    # print(loopnum)
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

def seq_update_book(W, input_pattern, nodes):
    output = np.zeros(nodes)
    order = np.arange(nodes)
    np.random.shuffle(order)
    for i in order:
        if np.sum(W[i,:]*input_pattern)>=0:
            output[i] = 1
        else:
            output[i] = -1
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


def main():


    for i in range(1, 301):
        original_patterns = random_pattern(i, 100)

    nodes = len(original_patterns[0])
    percentage = 10 #noise percentage

    ################
    #unbiased with diagonal

    # bias the training patterns
    bias_percentage = 0
    train_patterns = np.zeros((np.shape(original_patterns)))
    for i in range(original_patterns.shape[0]):
        train_patterns[i] = make_bias(original_patterns[i], bias_percentage)

    # add noise to the patterns that need to be recognised
    noisy_patterns = np.zeros((np.shape(train_patterns)))
    for i in range(noisy_patterns.shape[0]):
        noisy_patterns[i] = make_noise(train_patterns[i], percentage)

    # fig.suptitle("Synchronous update")
    capacity_percentage = []
    for i in range(train_patterns.shape[0]):
        patterns = train_patterns[0:i + 1]
        W = weight_matrix(nodes, patterns)
        # np.fill_diagonal(W, 0)
        saved = 0
        for j in range(i + 1):
            output2 = seq_update_book(W, noisy_patterns[j], nodes)
            diff = np.sum(abs(output2 - patterns[j]))
            if diff == 0:
                saved = saved + 1
        capacity_percentage.append(saved * 100 / (i + 1))
        print(capacity_percentage)

    plt.plot(np.arange(0, train_patterns.shape[0]), capacity_percentage, label = str(bias_percentage) + "% bias, WITH diagonal")

    ################
    # unbiased without diagonal

    # bias the training patterns
    bias_percentage = 0
    train_patterns = np.zeros((np.shape(original_patterns)))
    for i in range(original_patterns.shape[0]):
        train_patterns[i] = make_bias(original_patterns[i], bias_percentage)

    # add noise to the patterns that need to be recognised
    noisy_patterns = np.zeros((np.shape(train_patterns)))
    for i in range(noisy_patterns.shape[0]):
        noisy_patterns[i] = make_noise(train_patterns[i], percentage)

    # fig.suptitle("Synchronous update")
    capacity_percentage = []
    for i in range(train_patterns.shape[0]):
        patterns = train_patterns[0:i + 1]
        W = weight_matrix(nodes, patterns)
        np.fill_diagonal(W, 0)
        saved = 0
        for j in range(i + 1):
            output2 = seq_update_book(W, noisy_patterns[j], nodes)
            diff = np.sum(abs(output2 - patterns[j]))
            if diff == 0:
                saved = saved + 1
        capacity_percentage.append(saved * 100 / (i + 1))
        print(capacity_percentage)

    plt.plot(np.arange(0, train_patterns.shape[0]), capacity_percentage, label = str(bias_percentage) + "% bias, WO diagonal")

    ################
    # biased without diagonal

    for bias_percentage in [10, 20, 30]:
        train_patterns = np.zeros((np.shape(original_patterns)))
        for i in range(original_patterns.shape[0]):
            train_patterns[i] = make_bias(original_patterns[i], bias_percentage)

        # add noise to the patterns that need to be recognised
        noisy_patterns = np.zeros((np.shape(train_patterns)))
        for i in range(noisy_patterns.shape[0]):
            noisy_patterns[i] = make_noise(train_patterns[i], percentage)

        # fig.suptitle("Synchronous update")
        capacity_percentage = []
        for i in range(train_patterns.shape[0]):
            patterns = train_patterns[0:i + 1]
            W = weight_matrix(nodes, patterns)
            np.fill_diagonal(W, 0)
            saved = 0
            for j in range(i + 1):
                output2 = seq_update_book(W, noisy_patterns[j], nodes)
                diff = np.sum(abs(output2 - patterns[j]))
                if diff == 0:
                    saved = saved + 1
            capacity_percentage.append(saved * 100 / (i + 1))
            print(capacity_percentage)

        plt.plot(np.arange(0, train_patterns.shape[0]), capacity_percentage, label = str(bias_percentage) + "% bias, WO diagonal")


    plt.title("Capacity/patterns trained, with "+str(percentage)+"% of noise")
    plt.legend()
    plt.xlabel('Number of patterns')
    plt.ylabel('Capacity in percentage')

    #plt.xlim((0, 50))
    plt.show()




if __name__ == "__main__":
    main()
