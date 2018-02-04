import numpy as np
import matplotlib.pyplot as plt

def MackeyGlass(tau, beta, gamma, tmax, noise_var = 0):
    dataset = np.zeros(tmax)
    dataset[0] = 1.5 #first datapoint for t=0

    for t in range(1, tmax):
        if t-1-tau < 0 :
            dataset[t] = dataset[t - 1] - gamma * dataset[t - 1]
        else:
            dataset[t] = dataset[t-1] + ( beta * dataset[t-1-tau] ) / (1 + (dataset[t-1-tau]) ** 10) - gamma * dataset[t-1]

    if ( noise_var != 0 ):
        noise = np.random.normal(0, noise_var, tmax)
        dataset = dataset + noise

    return dataset

def main():
    plt.plot(MackeyGlass(25, 0.2, 0.1, 1200, False))
    plt.show()
    #print(MackeyGlass(25, 0.2, 0.1, 200))



if __name__ == "__main__":
    main()