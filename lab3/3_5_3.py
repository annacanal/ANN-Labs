import numpy as np
import matplotlib.pyplot as plt

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
    output = new_output
    output[output == -1] = 0
    iterations = loopnum-10
    return output, iterations

def random_pattern(row, column):
    y = np.random.uniform(-2, 2, (row, column))
    yy = np.sign(y)
    return yy

def weight_matrix(nodes, patterns):
    W = np.zeros((nodes, nodes))
    for k in range(patterns.shape[0]):
        W += (1 / nodes) * (np.outer(np.transpose(patterns[k]), patterns[k]))
    return W


def main():
    train_patterns = random_pattern(300, 100)
    nodes = 100

    for i in range(3,np.shape(train_patterns)[0], 10):
        W = weight_matrix(nodes, train_patterns[:i][:])

        correct = 0
        false = 0
        for j in range(i):
            output, iterations = sync_update(W, train_patterns[j])
            print(np.sum(abs(output - train_patterns[j])))
            if np.sum(abs(output - train_patterns[j])) < 50:
                correct += 1
            else:
                false += 1

        print("Train vectors:", i)
        print("Correct:", correct)
        print("False:", false)
        print("")

        # output, iterations = sync_update(W, noisy_patterns[i])

    print("2")



if __name__ == "__main__":
    main()
