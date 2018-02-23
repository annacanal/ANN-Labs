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

def random_pattern(number):
    np.random.uniform(-2, 2, (number, 10))

    for i in range(number):
        patterns = np.random.normal(0, 0.1, size)
        pattern_list.append(patterns)
    return pattern_list

def pattern_transform(pattern):
    new_pattern = pattern.reshape(32,32)
    return new_pattern

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
    output = bipolar_binary(new.reshape((-1, 1)))
    output = output.flatten()
    return output

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
            diffnum==0
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

def energy(weights, pattern):
    E = -1/2*(np.matmul(np.matmul(pattern, weights),np.transpose(pattern)))
    return E

def main():
    patterns_matrix = read_pictData()
    nodes = len(patterns_matrix[0])

    new_pattern_matrix = random_pattern(3)
    print(new_pattern_matrix)

    # # plot p1, p2, p3, p11 and p22
    # fig = plt.figure()
    # # p1
    # ax1 = fig.add_subplot(231)
    # ax1.imshow(pattern_transform(patterns_matrix[0]))
    # #ax1.title("p1")
    # ax1.set_title("p1")
    # # p2
    # ax2 = fig.add_subplot(232)
    # ax2.imshow(pattern_transform(patterns_matrix[1]))
    # ax2.set_title("p2")
    # # p3
    # ax3 = fig.add_subplot(233)
    # ax3.imshow(pattern_transform(patterns_matrix[2]))
    # ax3.set_title("p3")
    # # p4
    # ax6 = fig.add_subplot(234)
    # ax6.imshow(pattern_transform(patterns_matrix[3]))
    # ax6.set_title("p4")
    # # p11 and p22
    # ax4 = fig.add_subplot(235)
    # ax4.imshow(pattern_transform(patterns_matrix[9]))
    # ax4.set_title("p11 (degraded version of p1)")
    # # p11 and p22
    # ax5 = fig.add_subplot(236)
    # ax5.imshow(pattern_transform(patterns_matrix[10]))
    # ax5.set_title("p22 (mix of p2 and p3)")
    # plt.show()
    # #--------------------------------

    #train with p1, p2, p3 and p4
    train_patterns = np.concatenate(([patterns_matrix[0]],))
    W = weight_matrix(nodes, train_patterns)

    output = calc_activations(W,  patterns_matrix[9]) #check the p11 (which is the 10)
    E = energy(W, output)
    print(E)
    # fig = plt.figure()
    # fig.suptitle("Synchronous update")
    # ax1 = fig.add_subplot(121)
    # ax1.imshow(pattern_transform(output))
    # ax1.set_title("Recovered from p11\n" + str(iterations)+" iterations")
    # plt.show()


    # performance = []
    # train = []
    
    # for i in range(9):
    #     train_data = train.append(patterns_matrix[i])
    #     W = weight_matrix(nodes, train_data)
    #     output = calc_activations(W,  patterns_matrix[9])
    #     E = energy(W,output)
    #     performance.append(E)
    # print(performance)
    
    

if __name__ == "__main__":
    main()
