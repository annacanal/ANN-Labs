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
    W = np.zeros((nodes, nodes))
    for k in range(len(patterns)):
        W += (1 / nodes) * (np.outer(np.transpose(patterns[k]), patterns[k]))
    return W


def sync_update(W, input_pattern):
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
   # print("iterations:")
   # print(loopnum)
    output = new_output
    output[output == -1] = 0
    iterations = loopnum-10
    return output, iterations


def seq_update(W, input_pattern):
    output= input_pattern
    #Pick a random number between 1 and 1024
    idx = np.random.randint(1, 1024)
    output[idx] = np.sum(W[idx]*input_pattern.T)
    output[output >= 0] = 1
    output[output < 0] = -1
    return output

def make_noise(original_pattern, percentage):
    noisy_pattern = np.copy(original_pattern)
    #how many bits do we need to flip
    elements_num = noisy_pattern.size * percentage / 100
    elements_num = int(elements_num)
    #select that many random indexes
    indexes = np.random.choice(original_pattern.size, elements_num, replace=False)
    # flip the bit
    for i, idx in enumerate(indexes):
        noisy_pattern[idx] = -1 if noisy_pattern[idx] == 1 else 1

    return noisy_pattern


def main():
    patterns_matrix = read_pictData()
    nodes=1024

    # plot p1, p2, p3
    fig = plt.figure()
    # p1
    ax1 = fig.add_subplot(131)
    ax1.imshow(pattern_transform(patterns_matrix[0]))
    #ax1.title("p1")
    ax1.set_title("p1")
    # p2
    ax2 = fig.add_subplot(132)
    ax2.imshow(pattern_transform(patterns_matrix[1]))
    ax2.set_title("p2")
    # p3
    ax3 = fig.add_subplot(133)
    ax3.imshow(pattern_transform(patterns_matrix[2]))
    ax3.set_title("p3")


    #train with p1, p2, p3
    train_patterns = [patterns_matrix[0], patterns_matrix[1], patterns_matrix[2]]
    W = weight_matrix(nodes, train_patterns)


    for percentage in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:

        #make noise
        noisy1 = make_noise(train_patterns[0], percentage)
        noisy2 = make_noise(train_patterns[1], percentage)
        noisy3 = make_noise(train_patterns[2], percentage)

        ##################### Synchronous update ################################
        output1, iterations = sync_update(W, noisy1)
        output2, iterations = sync_update(W, noisy2)
        output3, iterations = sync_update(W, noisy3)


        ######################### OUTPUTS #####################################

        # plot outputs
        fig = plt.figure()
        fig.suptitle("Synchronous update")
        ax1 = fig.add_subplot(131)
        ax1.imshow(pattern_transform(output1))
        ax1.set_title("Recovered from p1\n with" + str(percentage) + "% noise")
        ax2 = fig.add_subplot(132)
        ax2.imshow(pattern_transform(output2))
        ax2.set_title("Recovered from p2\n with" + str(percentage) + "% noise")
        ax3 = fig.add_subplot(133)
        ax3.imshow(pattern_transform(output3))
        ax3.set_title("Recovered from p3\n with" + str(percentage) + "% noise")
        plt.show()




if __name__ == "__main__":
    main()
