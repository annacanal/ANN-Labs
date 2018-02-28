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
    # for i in range(number):
    #     patterns = np.random.normal(0, 0.1, size)
    #     pattern_list.append(patterns)
    # return pattern_list


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

def weight_matrix_zeroDiag(nodes, patterns):
    W = np.zeros([nodes, nodes])
    for k in range(len(patterns)):
        W += (1 / nodes) * (np.outer(np.transpose(patterns[k]), patterns[k]))
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if (j==i):
                W[i][j] = 0

    return W


def update(W, input_pattern,idx):
    #output = input_pattern
    output = np.sum(W * input_pattern.T)
    output[output>= 0] = 1
    output[output < 0] = -1
    new_output = output
    return new_output




def sync_update(W, input_pattern):
    old_output = input_pattern
    diffnum = 0
    loopnum = 0
    while diffnum < 10 and loopnum < 1000000:
    #for i in range(5):
        new_output = np.sum(W * old_output, axis=1)
        new_output[new_output >= 0] = 1
        new_output[new_output < 0] = -1
        diff = np.sum(abs(old_output - new_output))
        if diff == 0:
            diffnum += 1
        else:
            diffnum == 0
        old_output = new_output
        loopnum += 1
    # print("iterations:")
    # print(loopnum)
    output = new_output
    iterations = 1#loopnum - 10
    return output, iterations


def seq_update(W, input_pattern):
    output = input_pattern
    # Pick a random number between 1 and 1024
    idx = np.random.randint(1, 1024)
    output[idx] = np.sum(W[idx] * input_pattern.T)
    output[output >= 0] = 1
    output[output < 0] = -1
    return output


def energy(weights, pattern):
    E = -1 / 2 * (np.matmul(np.matmul(pattern, weights), np.transpose(pattern)))
    return E


def random_pattern(row, column):
    y = np.random.uniform(-2, 2, (row, column))
    yy = np.sign(y)
    return yy


def main():
 
    patterns_matrix = read_pictData()
    nodes = len(patterns_matrix[0])
    P,N = patterns_matrix.shape
    all_patterns = patterns_matrix
    # train with p1, p2, p3 and p4
    train_patterns = np.concatenate(([patterns_matrix[0]], [patterns_matrix[1]]))
    # print(train_patterns)
    output=[]
    capacity_percentage=[]

    fig = plt.figure()
    fig.suptitle("Synchronous update")
    E=[]
    for i in range(9):
        patterns = all_patterns[0:i]
        W = weight_matrix(nodes, patterns)
        saved = 0
        output2=0
        for j in range(i):
            output2,it = sync_update(W,  patterns[j])  # check the p10
          #  output = update(W, patterns[j])  # check the p10
            diff = np.sum(abs(output2 - patterns[j]))
            if diff == 0:
                saved = saved + 1
        capacity_percentage.append(saved * 100 / (i+1))
    plt.title("Capacity/patterns trained")
    plt.plot(np.arange(0,9), capacity_percentage)
    plt.show()

    ########### Fer random patterns

    for i in range(1, 200, 10):
        train_patterns = random_pattern(i, 1024)
    print(train_patterns.shape)

    fig = plt.figure()
  #  fig.suptitle("Synchronous update")
    capacity_percentage=[]
    for i in range(train_patterns.shape[0]):
        patterns = train_patterns[0:i]
        W = weight_matrix(nodes, patterns)
        saved = 0
        output2 = 0
        for j in range(i):
            output2, it = sync_update(W, patterns[j])  # check the p10
            #  output = update(W, patterns[j])  # check the p10
            diff = np.sum(abs(output2 - patterns[j]))
            if diff == 0:
                saved = saved + 1
        capacity_percentage.append(saved * 100 / (i + 1))
    plt.title("Capacity/patterns trained")
    plt.plot(np.arange(0, train_patterns.shape[0]), capacity_percentage)
    plt.show()



    # for i in range(4):
    #     pos = 2 + i
    #     saved=0
    #     pat = patterns_matrix[pos]  # .reshape(1,1024)
    #     new_train_patterns = np.vstack((train_patterns, pat))
    #     W = weight_matrix(nodes, new_train_patterns)
    #     W1 = weight_matrix_zeroDiag(nodes, new_train_patterns)
    #     for j in range(new_train_patterns.shape[1]):
    #         output = sync_update(W, patterns_matrix[9])  # check the p10
    #      #output,it = sync_update(W,  patterns_matrix[9])
    #         diff = np.sum(abs(patterns_matrix[9] - output))
    #         if diff == 0:
    #             saved = saved + 1
    #     capacity_percentage.append(saved * 100 / (pos + 1))
    #     E.append(energy(W, output))
    #     # Plot
    #     n = "14" + str(i + 1)
    #     ax = fig.add_subplot(n)
    #     ax.imshow(pattern_transform(output))
    #     ax.set_title(str(pos + 1) + " patterns")
    #     train_patterns = new_train_patterns
    #     plt.show()
    # plt.show()
    # plt.title("Energy/patterns trained")
    # plt.plot(np.arange(3, 7), E)
    # plt.show()


    # for i in range(9):
    #     train_data = train.append(patterns_matrix[i])
    #     W = weight_matrix(nodes, train_data)
    #     output = calc_activations(W,  patterns_matrix[9])
    #     E = energy(W,output)
    #     performance.append(E)
    # print(performance)


if __name__ == "__main__":
    main()
