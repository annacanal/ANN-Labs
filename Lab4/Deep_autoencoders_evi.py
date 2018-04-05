import numpy as np
import matplotlib.pyplot as plt
import csv
from keras.layers import Input, Dense
from keras.models import Model
import tensorflow as tf
from lab4.lab4_ae import load_data, sample_image, print_image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from math import sqrt


def train_DeepAutoencoder(train_data, train_labels,  num_epochs, num_hidden_nodes):

   ##########define the autoencoder############
    input_img = Input(shape=(train_data.shape[1],))
    num_layers = len(num_hidden_nodes)

    intermediate_models = []

    #configure network layers
    if (num_layers==1):
        encoded = Dense(num_hidden_nodes[0], activation='relu')(input_img)
        decoded = Dense(train_data.shape[1], activation='sigmoid')(encoded)

    elif (num_layers==2):
        encoded = Dense(num_hidden_nodes[0], activation='relu')(input_img)
        encoded = Dense(num_hidden_nodes[1], activation='relu')(encoded)

        decoded = Dense(num_hidden_nodes[0], activation='relu')(encoded)
        decoded = Dense(train_data.shape[1], activation='sigmoid')(decoded)

    else:
        encoded = Dense(num_hidden_nodes[0], activation='relu')(input_img)
        encoded = Dense(num_hidden_nodes[1], activation='relu')(encoded)
        encoded = Dense(num_hidden_nodes[2], activation='relu')(encoded)
        #encoded = Dense(num_hidden_nodes[2], activation='relu')(encoded) ###


        decoded = Dense(num_hidden_nodes[1], activation='relu')(encoded)
        decoded = Dense(num_hidden_nodes[0], activation='relu')(decoded)
        decoded = Dense(train_data.shape[1], activation='sigmoid')(decoded)
       # decoded = Dense(train_data.shape[1], activation='softmax')(decoded)  ####

    #configure model and train
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.fit(train_data, train_data, epochs=num_epochs, batch_size=256, shuffle=True, verbose=0) #class_weight


   # network output

   #define encoder and decoder model
    encoder = Model(input_img, encoded)
    encoded_input = Input(shape=(num_hidden_nodes[-1],))
    if(num_layers==3):
        #deco = autoencoder.layers[-4](encoded_input)
        deco = autoencoder.layers[-3](encoded_input)
        deco = autoencoder.layers[-2](deco)
        deco = autoencoder.layers[-1](deco)

        decoder = Model(encoded_input, deco)

        encoded_imgs = encoder.predict(train_data)
        decoded_imgs = decoder.predict(encoded_imgs)

    elif(num_layers==2):
        deco = autoencoder.layers[-2](encoded_input)
        deco = autoencoder.layers[-1](deco)

        decoder = Model(encoded_input, deco)

        encoded_imgs = encoder.predict(train_data)
        decoded_imgs = decoder.predict(encoded_imgs)

    else:
        deco = autoencoder.layers[-1](encoded_input)


        decoder = Model(encoded_input, deco)

        encoded_imgs = encoder.predict(train_data)
        decoded_imgs = decoder.predict(encoded_imgs)




    #decoded_imgs = autoencoder.predict(train_data)

    #print weights in final layer
    ae_weights = decoder.get_weights()
    num_nodes = ae_weights[0].shape[0]
    final_layer_dim = num_hidden_nodes[-1]

    weight_dim = int(sqrt(ae_weights[0].shape[1]))

    plt.figure(figsize=(20, 20))
    for i in range(num_nodes):
        weights_node = ae_weights[0][i]
        plt.subplot( weight_dim,  weight_dim, i + 1)
        plt.imshow(weights_node.reshape((weight_dim, weight_dim)), cmap=plt.cm.RdBu,
                   interpolation='nearest', vmin=-2.5, vmax=2.5)
        plt.axis('off')
    plt.suptitle('%d components extracted by Autoencoder' % final_layer_dim, fontsize=16)
    plt.savefig('Figures/AE_Final_%d.png' % final_layer_dim)
    plt.show()

    #get output from intermediate layers (bottom up) for a sample of images
    #intermediate_results = []
    samples = sample_image(train_data, train_labels) #sample one image per digit

    for i in range(num_layers):
        layer_model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(index=i).output)
        output = layer_model.predict(samples)
        #intermediate_results.append(output)
        print_image(samples, output, num_layers)
        print("Layer: ", i)
        print("output dim ", output.shape)
        print('-----------------')

    return decoded_imgs

#print a set of input and output images (one for each digit)
def print_image(input_imgs, decoded_imgs, num_layers):

    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 5))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(input_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        decoded_dim = int(sqrt(decoded_imgs.shape[1]))
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(decoded_dim, decoded_dim))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.suptitle('Overall Layers %d ' % num_layers, fontsize=16)
    plt.savefig('Figures/DeepAE_%d.png' % num_layers)
    plt.show()


def MLP_lastLayer(data, num_hidden_nodes):

    train_data = np.asmatrix(data[0])
    train_labels = np.array(data[1])
    test_data = np.asmatrix(data[2])
    test_labels = np.array(data[3])
    estimated_labels = np.zeros(test_labels.shape)

    mlp_class =  MLPClassifier(hidden_layer_sizes=(num_hidden_nodes,),solver='sgd',activation='relu',
                                verbose=False,random_state=0,early_stopping=True)

    mlp_class.fit(train_data, train_labels)

    for k, input_img in enumerate(test_data):
        estimated_labels[k] =  mlp_class.predict(input_img)

    accuracy = 1 - np.count_nonzero(np.abs(estimated_labels-test_labels))/test_labels.shape[0]

    print('\nCLASSIFICATION ERRORS')
    print(classification_report(test_labels, mlp_class.predict(test_data)))

    return accuracy

def logisticReg_lastlayer(data, num_epochs):

    train_data = data[0]
    train_labels = np.resize(data[1], (data[1].size,))
    test_data = data[2]
    test_labels = np.resize(data[1], (data[3].size,))

    # evaluate using Logistic Regression
    logistic = LogisticRegression(C=10.0,random_state=0,max_iter=num_epochs,verbose=0)
    logistic.fit(train_data, train_labels)

    dim = train_data.shape[1]
    print('\nCLASSIFICATION ERRORS')
    print(classification_report(test_labels, logistic.predict(test_data)))
    #accuracy = logistic.score(test_data, test_labels)



def test_deepAutoencoder(output_layer): #0:MLP, 1:Logistic Regression

    data = load_data()

    train_data = data[0]
    train_labels = data[1]
    test_data = data[2]
    test_labels = data[3]

    # Training parameters
    num_epochs = 500
    #hidden_nodes = [11**2, 10**2, 9**2]
    hidden_nodes = [625, 576, 529]

    for i in range(0, 1):
        #Deep Autoencoder
        #output_images = train_DeepAutoencoder(train_data, train_labels, num_epochs, hidden_nodes[0:i])
        #num_layers = len(hidden_nodes[0:i])
        output_images = train_data
        num_layers = 0

        if (output_layer == 0):
            # Output layer (MLPClassifier)
            mlp_hidden_nodes = 100

            accuracy = MLP_lastLayer((output_images, train_labels, test_data, test_labels), mlp_hidden_nodes)
            print('Accuracy of the DNN when using %d hidden layers = %.3f (MLP)' % (num_layers, accuracy))
        else:
            # Output layer Logistic Regression
            logisticReg_lastlayer((output_images, train_labels, test_data, test_labels), num_epochs)



test_deepAutoencoder(0)