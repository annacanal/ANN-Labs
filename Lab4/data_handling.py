import numpy as np
import csv, os, operator
from io import BytesIO

def read_train_dataset():
    train = np.zeros((8000,784))
    train_targets=np.zeros((8000,1))
    # Read data train file
    with open("binMNIST_data/bindigit_trn.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        i=0
        for row in reader:
            row = list(map(int, row))
            for c in range(len(row)):
                col = row[c]
                train[i][c]= col
            i=i+1
    #Read targets train file
    with open("binMNIST_data/targetdigit_trn.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        i=0
        for row in reader:
            row = list(map(int, row))
            r= row[0]
            train_targets[i][0]= r
            i=i+1
    return train, train_targets


def read_test_dataset():
    test = np.zeros((2000, 784))
    test_targets = np.zeros((2000, 1))
    # Read data train file
    with open("binMNIST_data/bindigit_tst.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        i = 0
        for row in reader:
            row = list(map(int, row))
            for c in range(len(row)):
                col = row[c]
                test[i][c] = col
            i = i + 1
    # Read targets train file
    with open("binMNIST_data/targetdigit_tst.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        i = 0
        for row in reader:
            row = list(map(int, row))
            r = row[0]
            test_targets[i][0] = r
            i = i + 1
    return test, test_targets



def main():
    train,train_targets = read_train_dataset()
    print("train targets:")
    print(train_targets)
    print("train data:")
    print(train)
    test, test_targets = read_test_dataset()
    print("test targets:")
    print(test_targets)
    print("test data:")
    print(test)

if __name__ == "__main__":
    main()