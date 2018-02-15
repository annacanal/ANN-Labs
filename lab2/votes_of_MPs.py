import numpy as np
import matplotlib.pyplot as plt

def get_data_matrix():
    doc = ["data_lab2/mpparty.dat","data_lab2/mpsex.dat","data_lab2/mpdistrict.dat"]
    matrices = []
    for i in range(len(doc)):
        with open(doc[i], "r") as file:
            # Read the whole file at once
            data = file.readlines()
        data_matrix = np.zeros((1, 349))
        for i in range(data_matrix.shape[0]):
            for j in range(data_matrix.shape[1]):
                position = i*data_matrix.shape[1] + j
                data_matrix[i][j]= data[position]
        matrices.append(data_matrix)
    party_matrix = matrices[0]
    gender_matrix = matrices[1]
    district_matrix = matrices[2]

    with open("data_lab2/votes.dat", "r") as binary_file:
        # Read the whole file at once
        data = binary_file.read()
    data2 = data.split(",")
    vote_matrix = np.zeros((349,31))
    for i in range(vote_matrix.shape[0]):
        for j in range(vote_matrix.shape[1]):
            position = i*vote_matrix.shape[1] + j
            vote_matrix[i][j]= data2[position]

    return party_matrix, gender_matrix, district_matrix, vote_matrix

def get_party_color(party):
    if party == 0:
        color = 'grey'
        legend = 'no party'
    elif party == 1:
        color = 'red'
        legend = 'm'
    elif party == 2:
        color = 'firebrick'
        legend = 'fp'
    elif party == 3:
        color = 'skyblue'
        legend = 's'
    elif party == 4:
        color = 'dodgerblue'
        legend = 'v'
    elif party == 5:
        color = 'purple'
        legend = 'mp'
    elif party == 6:
        color = 'maroon'
        legend = 'kd'
    elif party == 7:
        color = 'lightcoral'
        legend = 'c'
    else:
        print(party, "WRONGPARTY")
        exit(1)

    return(color, legend)

def get_gender_color(gender):
    if gender == 0:
        color = 'skyblue'
        marker = '^'
        label = 'male'
    elif gender == 1:
        color = 'lightcoral'
        marker = 'v'
        label = 'female'
    else:
        print(gender, "WRONGGENDER")
        exit(1)

    return(color, marker, label)

def get_district_color(districts, dstr):
    #we will get greyscale colors between 0 and 1
    color = str(float(dstr)/float(max(districts)))

    return(color)


def find_bmu(t, net):
    """
        Find the best matching unit for a given vector, t, in the SOM
        Returns: a (bmu, bmu_idx) tuple where bmu is the high-dimensional BMU
                 and bmu_idx is the index of this vector in the SOM
    """
    bmu_idx = [0,0]
    # set the initial minimum distance to a huge number
    min_dist = np.iinfo(np.int).max
    # calculate the high-dimensional distance between each neuron and the input
    for x in range(net.shape[0]):
        for y in range(net.shape[1]):
            w = net[x][y]#.reshape(1,84)
            # don't bother with actual Euclidean distance, to avoid expensive sqrt operation
            sq_dist = np.sum(np.square(w - t))
            if sq_dist < min_dist:
                min_dist = sq_dist
                #bmu_idx = np.array([x, y])
                bmu_idx = [x,y]
    # get vector corresponding to bmu_idx
    #bmu = net[bmu_idx[0], bmu_idx[1], :].reshape(m, 1)
    bmu = net[bmu_idx[0]][bmu_idx[1]]#.reshape(84,1)
    # return the (bmu, bmu_idx) tuple
    return (bmu, bmu_idx)

def decay_radius(initial_radius, i, time_constant):
    return initial_radius * np.exp(-i / time_constant)

def decay_learning_rate(initial_learning_rate, i, epochs):
    return initial_learning_rate * np.exp(-i / epochs)

def calculate_influence(distance, radius):
    return np.exp(-distance / (2* (radius**2)))


def main():
    party_matrix, gender_matrix, district_matrix, votes_data = get_data_matrix()
    epochs = 20
    init_learning_rate = 0.2
    # weight matrix (i.e. the SOM) needs to be one m-dimensional vector for each neuron in the SOM
    net = np.random.random((10,10,31))
    # initial neighbourhood radius
    init_radius = 5
    # radius decay parameter
    time_constant = epochs / np.log(init_radius)

    # Learning
    for i in range(epochs):
        r = decay_radius(init_radius, i, time_constant)
        l = decay_learning_rate(init_learning_rate, i, epochs)
        # for each MP
        for j in range((votes_data.shape[0])):
            row_p = votes_data[j][:]
            bmu, bmu_idx = find_bmu(row_p, net)


            # now we know the BMU, update its
            # weight vector to move closer to input
            # and move its neighbours in 2-D space closer
            # by a factor proportional to their 2-D distance from the BMU
            for x in range(net.shape[0]):
                for y in range(net.shape[1]):
                    w = net[x][y]#.reshape(1, 84)
                    # get the 2-D distance (again, not the actual Euclidean distance)
                    w_dist = np.sum(np.square(np.array([x,y]) - np.array(bmu_idx)))
                    # if the distance is within the current neighbourhood radius
                    if w_dist <= r ** 2:
                        # calculate the degree of influence (based on the 2-D distance)
                        influence = calculate_influence(w_dist, r)
                        # now update the neuron's weight using the formula:
                        # new w = old w + (learning rate * influence * delta)
                        # where delta = input vector (t) - old w
                        new_w = w + (l * influence * (row_p - w))

                        # commit the new weight
                        net[x][y] = new_w

    #display the results for parties
    ax = plt.subplot(111)
    plt.title("Votes according to party")
    parties = set(party_matrix[0][:])
    parties = list(parties)
    for prt in range(len(parties)):
        posx = []
        posy = []
        for i in range(votes_data.shape[0]):
            partidx = parties.index(party_matrix[0][i])
            if partidx == prt:
                row_p = votes_data[i][:]
                bmu, bmu_idx = find_bmu(row_p, net)

                posx.append(bmu_idx[0])
                posy.append(bmu_idx[1])
        clr,lbl = get_party_color(prt)
        ax.scatter(posx, posy, color = clr, label = lbl, alpha=0.7)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc = 'center left', bbox_to_anchor=(1, 0.5))

    plt.show()
    plt.clf()


    # display the results for gender
    ax = plt.subplot(111)
    plt.title("Votes according to gender")
    genders = list(set(gender_matrix[0][:]))
    for gnd in range(len(genders)):
        posx = []
        posy = []
        for i in range(votes_data.shape[0]):
            gndidx = genders.index(gender_matrix[0][i])
            if gndidx == gnd:
                row_p = votes_data[i][:]
                bmu, bmu_idx = find_bmu(row_p, net)

                posx.append(bmu_idx[0])
                posy.append(bmu_idx[1])
        clr, mrk, lbl = get_gender_color(gnd)
        ax.scatter(posx, posy, color=clr, marker = mrk, label = lbl, alpha=0.7)
    ax.legend(loc = 'center', bbox_to_anchor=(0.9, 1.07))
    plt.show()
    plt.clf()

    # display the results for district
    plt.title("Votes according to district")
    districts = list(set(district_matrix[0][:]))
    for dstr in range(len(districts)):
        posx = []
        posy = []
        for i in range(votes_data.shape[0]):
            dstridx = districts.index(district_matrix[0][i])
            if dstridx == dstr:
                row_p = votes_data[i][:]
                bmu, bmu_idx = find_bmu(row_p, net)

                posx.append(bmu_idx[0])
                posy.append(bmu_idx[1])
        clr = get_district_color(districts, districts[dstr])
        plt.scatter(posx, posy, color=clr)
    plt.show()
    plt.clf()




if __name__ == "__main__":
     main()
