import numpy as np
import matplotlib.pyplot as plt

def get_data_matrix():
    with open("data_lab2/cities.dat", "r") as file:
        # Read the whole file at once
        data = file.readlines()
    #print(data)
    city_matrix = np.zeros((10,2))
    for i in range(len(data)):
        data[i].split(",")
        city_matrix[i][0]= data[i][0:6]
        city_matrix[i][1] = data[i][8:14]
        # print(data[i])
        # print(city_matrix[i][0])
        # print(city_matrix[i][1])
    return city_matrix


def find_bmu(t, net):
    bmu_idx = []
    # set the initial minimum distance to a huge number
    min_dist = np.iinfo(np.int).max
    # calculate the high-dimensional distance between each neuron and the input
    for x in range(net.shape[0]):
        #for y in range(net.shape[1]):
        w = net[x].reshape(1,2)
        # don't bother with actual Euclidean distance, to avoid expensive sqrt operation
        sq_dist = np.sum((w - t) ** 2)
        if sq_dist < min_dist:
            min_dist = sq_dist
            #bmu_idx = np.array([x, y])
            bmu_idx = x
    # get vector corresponding to bmu_idx
    #bmu = net[bmu_idx[0], bmu_idx[1], :].reshape(m, 1)
    bmu = net[bmu_idx].reshape(1, 2)
    # return the (bmu, bmu_idx) tuple
    return (bmu, bmu_idx)

def decay_radius(initial_radius, i, time_constant):
    return initial_radius * np.exp(-i / time_constant)

def decay_learning_rate(initial_learning_rate, i, epochs):
    return initial_learning_rate * np.exp(-i / epochs)

def calculate_influence(distance, radius):
    return np.exp(-distance / (2* (radius**2)))

def main():
    cities_data_matrix = get_data_matrix()
    print(cities_data_matrix)
    #epochs = 5
    # weight matrix (i.e. the SOM) needs to be one m-dimensional vector for each neuron in the SOM
    net = np.random.random((10, 2))

    #Learning
    epochs_arr = np.arange(1,15,1)
    for epochs in epochs_arr:
        init_learning_rate = 0.2
        # initial neighbourhood radius
        init_radius = 2
        # radius decay parameter
        time_constant = epochs / np.log(init_radius)
        for i in range(epochs):
            for j in range(len(cities_data_matrix)):
                row_p = cities_data_matrix[j][:]
                bmu, bmu_idx = find_bmu(row_p, net)

                r = decay_radius(init_radius, i, time_constant)
                l = decay_learning_rate(init_learning_rate, i, epochs)
                # now we know the BMU, update its
                # weight vector to move closer to input
                # and move its neighbours in 2-D space closer
                # by a factor proportional to their 2-D distance from the BMU
                for x in range(net.shape[0]):
                #    for y in range(net.shape[1]):
                    w = net[x].reshape(1, 2)
                    # get the 2-D distance (again, not the actual Euclidean distance)
                    w_dist = np.sum((x - bmu_idx) ** 2)
                    # if the distance is within the current neighbourhood radius
                    if w_dist <= r**2:
                        # calculate the degree of influence (based on the 2-D distance)
                        influence = calculate_influence(w_dist, r)
                        # now update the neuron's weight using the formula:
                        # new w = old w + (learning rate * influence * delta)
                        # where delta = input vector (t) - old w
                        new_w = w + (l * influence * (row_p - w))
                        # commit the new weight
                        net[x] = new_w[0].reshape(1,2)

        # Find index for species
        pos = []
        idx_cities = []
        for i in range(len(cities_data_matrix)):
            row_p = cities_data_matrix[i][:]
            bmu, bmu_idx = find_bmu(row_p, net)
            idx_cities.append(i)
            pos.append(bmu_idx)
        ordered_pos = sorted(zip(pos, idx_cities))
        #print(ordered_pos)

        cities1 = []
        cities2 = []
        for i in range(len(cities_data_matrix)):
            pos = ordered_pos[i][1]
            #print(pos)
            cities1.append(cities_data_matrix[pos][0])
            cities2.append(cities_data_matrix[pos][1])
            #plt.plot(cities_data_matrix[pos][0],cities_data_matrix[pos][1])
        #print(cities[][0])
        #for i in range(10):
        plt.scatter(cities1,cities2, color='r')
        plt.plot(cities1, cities2)
        plt.title("City tour with epochs = "+str(epochs))
        plt.show()

if __name__ == "__main__":
    main()
