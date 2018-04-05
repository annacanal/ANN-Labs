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

def plot_digits(dataset, predictions, title):
    # ---------Plotting digits: ---------------------
    example_indexs = [18, 3, 7, 0, 2, 1, 14, 8, 6, 5]
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(dataset[example_indexs[i]].reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(predictions[example_indexs[i]].reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # np.save('test_3_layers_75epochs_0.3lr.npy', hist.history)
    plt.title(title)
    plt.show()
    plt.close()


def mel_training_2(train, test, epochs, batch_size):
    input_img = Input(shape=(784,))
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)

    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(784, activation='sigmoid')(decoded)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoencoder.fit(train, train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(test, test))

    

def mel_training(my_epochs, layers, train, test, train_targets, test_targets):
    # Layer by layer pretraining Models

    # Layer 1
    input_img = Input(shape=(784,))
    encoded1 = Dense(600, activation='sigmoid', kernel_initializer='random_normal', bias_initializer='zeros')(input_img)
    decoded1 = Dense(784, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros')(encoded1)

    autoencoder1 = Model(input_img, decoded1)
    encoder1 = Model(input_img, encoded1)

    # Layer 2
    encoded1_input = Input(shape=(600,))
    encoded2 = Dense(400, activation='sigmoid', kernel_initializer='random_normal', bias_initializer='zeros')(encoded1_input)
    decoded2 = Dense(600, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros')(encoded2)

    autoencoder2 = Model(encoded1_input, decoded2)
    encoder2 = Model(encoded1_input, encoded2)

    # Layer 3 - which we won't end up fitting in the interest of time
    encoded2_input = Input(shape=(400,))
    encoded3 = Dense(200, activation='sigmoid', kernel_initializer='random_normal', bias_initializer='zeros')(encoded2_input)
    decoded3 = Dense(400, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros')(encoded3)

    autoencoder3 = Model(encoded2_input, decoded3)
    encoder3 = Model(encoded2_input, encoded3)

    #not so deep Autoencoder

    encoded1_nda = Dense(600, activation='sigmoid', kernel_initializer='random_normal', bias_initializer='zeros')(input_img)
    encoded2_nda = Dense(400, activation='sigmoid', kernel_initializer='random_normal', bias_initializer='zeros')(encoded1_nda)
    decoded2_nda = Dense(600, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros')(encoded2_nda)
    decoded1_nda = Dense(784, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros')(decoded2_nda)

    not_so_deep_autoencoder = Model(input_img, decoded1_nda)


    # Deep Autoencoder
    encoded1_da = Dense(600, activation='sigmoid', kernel_initializer='random_normal', bias_initializer='zeros')(input_img)
    encoded2_da = Dense(400, activation='sigmoid', kernel_initializer='random_normal', bias_initializer='zeros')(encoded1_da)
    encoded3_da = Dense(200, activation='sigmoid', kernel_initializer='random_normal', bias_initializer='zeros')(encoded2_da)
    decoded3_da = Dense(400, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros')(encoded3_da)
    decoded2_da = Dense(600, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros')(decoded3_da)
    decoded1_da = Dense(784, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros')(decoded2_da)

    deep_autoencoder = Model(input_img, decoded1_da)

    sgd = optimizers.SGD(lr=0.5, momentum=0, decay=0, nesterov=False)

    autoencoder1.compile(optimizer=sgd, loss='mean_squared_error', metrics=['mae'])
    autoencoder2.compile(optimizer=sgd, loss='mean_squared_error', metrics=['mae'])
    autoencoder3.compile(optimizer=sgd, loss='mean_squared_error', metrics=['mae'])

    encoder1.compile(optimizer=sgd, loss='mean_squared_error', metrics=['mae'])
    encoder2.compile(optimizer=sgd, loss='mean_squared_error', metrics=['mae'])
    encoder3.compile(optimizer=sgd, loss='mean_squared_error', metrics=['mae'])

    not_so_deep_autoencoder.compile(optimizer=sgd, loss='mean_squared_error', metrics=['mae'])
    deep_autoencoder.compile(optimizer=sgd, loss='mean_squared_error', metrics=['mae'])

    autoencoder1.fit(train, train,
                     epochs=20,
                     batch_size=5,
                     validation_split=0.30,
                     shuffle=True,
                     verbose=2,)

    first_layer_code = encoder1.predict(train)
    autoencoder2.fit(first_layer_code, first_layer_code,
                     epochs=20,
                     batch_size=5,
                     validation_split=0.3,
                     shuffle=True,
                     verbose=2,)

    second_layer_code = encoder2.predict(first_layer_code)
    autoencoder3.fit(second_layer_code, second_layer_code,
                     epochs = 20,
                     batch_size = 5,
                     validation_split = 0.30,
                     shuffle = True,
                     verbose=2)

    predictions = autoencoder1.predict(test)
    plot_digits(test, predictions, "autoencoder 1")


    weights1 = autoencoder1.get_weights()
    weights2 = autoencoder2.get_weights()
    weights3 = autoencoder3.get_weights()



    ###############################################

    not_so_deep_autoencoder_weights = not_so_deep_autoencoder.get_weights()

    not_so_deep_autoencoder_weights[0] = weights1[0]
    not_so_deep_autoencoder_weights[1] = weights1[1]

    not_so_deep_autoencoder_weights[2] = weights2[0]
    not_so_deep_autoencoder_weights[3] = weights2[1]

    not_so_deep_autoencoder_weights[4] = weights2[2]
    not_so_deep_autoencoder_weights[5] = weights2[3]

    not_so_deep_autoencoder_weights[6] = weights1[2]
    not_so_deep_autoencoder_weights[7] = weights1[3]

    not_so_deep_autoencoder.set_weights(not_so_deep_autoencoder_weights)

    predictions = not_so_deep_autoencoder.predict(test)
    plot_digits(test, predictions, "two layers")

    ##############################################

    deep_autoencoder_weights = deep_autoencoder.get_weights()


    deep_autoencoder_weights[0] = weights1[0]
    deep_autoencoder_weights[1] = weights1[1]

    deep_autoencoder_weights[2] = weights2[0]
    deep_autoencoder_weights[3] = weights2[1]

    deep_autoencoder_weights[4] = weights3[0]
    deep_autoencoder_weights[5] = weights3[1]

    deep_autoencoder_weights[6] = weights3[2]
    deep_autoencoder_weights[7] = weights3[3]

    deep_autoencoder_weights[8] = weights2[2]
    deep_autoencoder_weights[9] = weights2[3]

    deep_autoencoder_weights[10] = weights1[2]
    deep_autoencoder_weights[11] = weights1[3]

    deep_autoencoder.set_weights(deep_autoencoder_weights)


    predictions = deep_autoencoder.predict(test)
    plot_digits(test, predictions, "3 layers")

def training(my_epochs, layers, input_img, train, test, train_targets, test_targets):
#-----------pre-training------------------------
    if layers == 3: 
        encoded = Dense(150, activation = 'relu')(input_img)
        encoded = Dense(120, activation = 'relu')(encoded)
        encoded = Dense(90, activation = 'relu')(encoded)

        decoded = Dense(120, activation = 'relu')(encoded)
        decoded = Dense(150, activation = 'relu')(decoded)
        decoded = Dense(784, activation='sigmoid')(decoded)

    elif layers == 2:
        encoded = Dense(150, activation = 'relu')(input_img)
        encoded = Dense(120, activation = 'relu')(encoded)

        decoded = Dense(150, activation = 'relu')(encoded)
        decoded = Dense(784, activation='sigmoid')(decoded)

    elif layers == 1:
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

def main():
    train, train_targets = data_handling.read_train_dataset()
    test, test_targets = data_handling.read_test_dataset()
    layers_list = 3
    my_epochs = 500
    # for layers in layers_list:
    mel_training(my_epochs, layers_list, train, test, train_targets, test_targets)

if __name__ == "__main__":
    main()