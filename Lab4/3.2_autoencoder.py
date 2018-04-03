from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers
import data_handling
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from keras.utils import np_utils
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report

def main():
    train, train_targets = data_handling.read_train_dataset()
    test, test_targets = data_handling.read_test_dataset()
    input_img = Input(shape=(784,))
    layers_list = 3
    my_epochs = 500
    # for layers in layers_list:
    training(my_epochs, layers_list, input_img, train, test, train_targets, test_targets)


def training(my_epochs, layers, input_img, train, test, train_targets, test_targets):
#-----------pre-training------------------------
    if layers == 3: 
        encoded = Dense(150, activation = 'relu')(input_img)
        encoded = Dense(120, activation = 'relu')(encoded)
        encoded = Dense(90, activation = 'relu')(encoded)

        decoded = Dense(120, activation = 'relu')(encoded)
        decoded = Dense(150, activation = 'relu')(decoded)
        decoded = Dense(784, activation='sigmoid')(decoded)

    if layers == 2: 
        encoded = Dense(150, activation = 'relu')(input_img)
        encoded = Dense(120, activation = 'relu')(encoded)

        decoded = Dense(150, activation = 'relu')(encoded)
        decoded = Dense(784, activation='sigmoid')(decoded)

    if layers == 1:
        encoded = Dense(150, activation = 'relu')(input_img)

        decoded = Dense(784, activation='sigmoid')(encoded)

    #-----------Autoencoder: 
    if layers != 0:
        autoencoder = Model(input_img, decoded)
        sgd = optimizers.SGD(lr=2, momentum=0.9, decay=0, nesterov=False)
        autoencoder.compile(optimizer=sgd, loss='mean_squared_error', metrics=['mae'])
        autoencoder.fit(train, train,
                        epochs=my_epochs,
                        batch_size= 256, #default
                        shuffle=True, verbose = 0,
                        validation_data=(test, test))

        #-----------Encoder:
        encoder = Model(input=input_img, output=encoded)
    
    #-----------Decoder
    if layers == 3: 
        encoded_input_1 = Input(shape=(90,))
        encoded_input_2 = Input(shape=(120,))
        encoded_input_3 = Input(shape=(150,))

        decoder_layer_1 = autoencoder.layers[-3]
        decoder_layer_2 = autoencoder.layers[-2]
        decoder_layer_3 = autoencoder.layers[-1]

        decoder_1 = Model(input = encoded_input_1, output = decoder_layer_1(encoded_input_1))
        decoder_2 = Model(input = encoded_input_2, output = decoder_layer_2(encoded_input_2))
        decoder_3 = Model(input = encoded_input_3, output = decoder_layer_3(encoded_input_3))

        encoded_imgs = encoder.predict(train)
        decoded_imgs = decoder_1.predict(encoded_imgs) 
        decoded_imgs = decoder_2.predict(decoded_imgs)
        decoded_imgs = decoder_3.predict(decoded_imgs)

    if layers == 2: 
        encoded_input_2 = Input(shape=(120,))
        encoded_input_3 = Input(shape=(150,))

        decoder_layer_2 = autoencoder.layers[-2]
        decoder_layer_3 = autoencoder.layers[-1]

        decoder_2 = Model(input = encoded_input_2, output = decoder_layer_2(encoded_input_2))
        decoder_3 = Model(input = encoded_input_3, output = decoder_layer_3(encoded_input_3))

        encoded_imgs = encoder.predict(train)
        decoded_imgs = decoder_2.predict(encoded_imgs)
        decoded_imgs = decoder_3.predict(decoded_imgs)

    if layers == 1: 
        encoded_input_3 = Input(shape=(150,))
        decoder_layer_3 = autoencoder.layers[-1]
        decoder_3 = Model(input = encoded_input_3, output = decoder_layer_3(encoded_input_3))

        encoded_imgs = encoder.predict(train)
        decoded_imgs = decoder_3.predict(encoded_imgs)

#--------- evaluate using Logistic Regression
    logistic = LogisticRegression(C=10.0,random_state=0,max_iter=my_epochs)
    
    if layers == 0: 
        logistic.fit(train, train_targets)
    else:
        logistic.fit(decoded_imgs, train_targets)

        # dim = train_data.shape[1]
    print('\nCLASSIFICATION ERRORS')
    print(classification_report(test_targets, logistic.predict(test)))

    if layers == 3:
        encoded_imgs = encoder.predict(test)
        decoded_imgs = decoder_1.predict(encoded_imgs) 
        decoded_imgs = decoder_2.predict(decoded_imgs)
        decoded_imgs = decoder_3.predict(decoded_imgs)

    if layers == 2: 
        encoded_imgs = encoder.predict(test)
        decoded_imgs = decoder_2.predict(encoded_imgs)
        decoded_imgs = decoder_3.predict(decoded_imgs)

    if layers == 1: 
        encoded_imgs = encoder.predict(test)
        decoded_imgs = decoder_3.predict(encoded_imgs)

#---------Plotting digits: ---------------------
    example_indexs = [18, 3, 7, 0, 2, 1, 14, 8, 6, 5]
    n = 10  
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(test[example_indexs[i]].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[example_indexs[i]].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
            
        # np.save('test_3_layers_75epochs_0.3lr.npy', hist.history)
    plt.show()

#---------Plotting weights: ---------------
    if layers == 3: 
        weights_3 = decoder_3.get_weights()[0]
        weights_2 = decoder_2.get_weights()[0]
        weights_1 = decoder_1.get_weights()[0]

        plt.figure(figsize=(10, 10))
        for i in range(150):
            plt.subplot(10, 15, i + 1)
            plt.imshow(weights_3[i].reshape((28, 28)), cmap=plt.cm.gray_r,
                        interpolation='nearest')
            plt.xticks(())
            plt.yticks(())
        plt.suptitle('150 components extracted by autoencoder layer 1', fontsize=16)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
        plt.show()

        plt.figure(figsize=(10, 10))
        for i in range(120):
            plt.subplot(10, 12, i + 1)
            plt.imshow(weights_2[i].reshape((10, 15)), cmap=plt.cm.gray_r,
                        interpolation='nearest')
            plt.xticks(())
            plt.yticks(())
        plt.suptitle('120 components extracted by autoencoder layer 2', fontsize=16)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
        plt.show()

        plt.figure(figsize=(10, 10))
        for i in range(90):
            plt.subplot(10, 10, i + 1)
            plt.imshow(weights_1[i].reshape((10, 12)), cmap=plt.cm.gray_r,
                        interpolation='nearest')
            plt.xticks(())
            plt.yticks(())
        plt.suptitle('90 components extracted by autoencoder layer 3', fontsize=16)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
        plt.show()

if __name__ == "__main__":
    main()