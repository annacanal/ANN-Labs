import numpy as np

def get_data_matrix():
    with open("data_lab2/mpsex.dat", "r") as file:
        # Read the whole file at once
        data = file.readlines()
    gender_matrix = np.zeros((1,349))
    for i in range(gender_matrix.shape[0]-1):
        for j in range(data_matrix.shape[1] - 1):
            position = i*data_matrix.shape[1] + j
            data_matrix[i][j]= data2[position]

    return data_matrix

# def get_animal_names():
#     with open("data_lab2/animalnames.txt", "r") as f:
#         # Read the whole file at once
#         names = f.readlines()
#     names = [x.strip() for x in names]
#     return names

# def find_bmu(t, net):
#     """
#         Find the best matching unit for a given vector, t, in the SOM
#         Returns: a (bmu, bmu_idx) tuple where bmu is the high-dimensional BMU
#                  and bmu_idx is the index of this vector in the SOM
#     """
#     bmu_idx = []
#     # set the initial minimum distance to a huge number
#     min_dist = np.iinfo(np.int).max
#     # calculate the high-dimensional distance between each neuron and the input
#     for x in range(net.shape[0]):
#         #for y in range(net.shape[1]):
#         w = net[x].reshape(1,84)
#         # don't bother with actual Euclidean distance, to avoid expensive sqrt operation
#         sq_dist = np.sum((w - t) ** 2)
#         if sq_dist < min_dist:
#             min_dist = sq_dist
#             #bmu_idx = np.array([x, y])
#             bmu_idx = x
#     # get vector corresponding to bmu_idx
#     #bmu = net[bmu_idx[0], bmu_idx[1], :].reshape(m, 1)
#     bmu = net[bmu_idx].reshape(84,1)
#     # return the (bmu, bmu_idx) tuple
#     return (bmu, bmu_idx)

# def decay_radius(initial_radius, i, time_constant):
#     return initial_radius * np.exp(-i / time_constant)

# def decay_learning_rate(initial_learning_rate, i, epochs):
#     return initial_learning_rate * np.exp(-i / epochs)

# def calculate_influence(distance, radius):
#     return np.exp(-distance / (2* (radius**2)))

# def main():
#     animals_data = get_data_matrix()
#     epochs = 20
#     init_learning_rate = 0.2
#     # weight matrix (i.e. the SOM) needs to be one m-dimensional vector for each neuron in the SOM
#     net = np.random.random((100, 84))
#     # initial neighbourhood radius
#     init_radius = 50
#     # radius decay parameter
#     time_constant = epochs / np.log(init_radius)
#     #Learning
#     for i in range(epochs):
#         for j in range(len(animals_data)):
#             row_p = animals_data[j][:]
#             bmu, bmu_idx = find_bmu(row_p, net)

#             r = decay_radius(init_radius, i, time_constant)
#             l = decay_learning_rate(init_learning_rate, i, epochs)
#             # now we know the BMU, update its
#             # weight vector to move closer to input
#             # and move its neighbours in 2-D space closer
#             # by a factor proportional to their 2-D distance from the BMU
#             for x in range(net.shape[0]):
#             #    for y in range(net.shape[1]):
#                 w = net[x].reshape(1,84)
#                 # get the 2-D distance (again, not the actual Euclidean distance)
#                 w_dist = np.sum((x - bmu_idx) ** 2)                                        
#                 # if the distance is within the current neighbourhood radius
#                 if w_dist <= r**2:
#                     # calculate the degree of influence (based on the 2-D distance)
#                     influence = calculate_influence(w_dist, r)
#                     # now update the neuron's weight using the formula:
#                     # new w = old w + (learning rate * influence * delta)
#                     # where delta = input vector (t) - old w
#                     new_w = w + (l * influence * (row_p - w))
#                     # commit the new weight
#                     net[x] = new_w[0].reshape(1,84)

#     # Find index for species
#     names = get_animal_names()
#     pos = []
#     idx_names = []
#     for i in range(len(animals_data)):
#         row_p = animals_data[i][:]
#         bmu, bmu_idx = find_bmu(row_p, net)
#         idx_names.append(i)
#         pos.append(bmu_idx)


#     ordered_pos = sorted(zip(pos, idx_names))
#     print(ordered_pos)
#     names_order=[]
#     for i in range(len(names)):
#         pos = ordered_pos[i][1]
#         names_order.append(names[pos])
#         print(names_order[i])
#     #print(names_order)

# if __name__ == "__main__":
#     main()
