from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers
import data_handling
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from keras.utils import np_utils

#-----------pre-training------------------------

input_img = Input(shape=(784,))

#3 hidden layers: 
encoded = Dense(150, activation = 'relu', kernel_initializer='random_normal', bias_initializer='zeros')(input_img)
encoded = Dense(120, activation = 'relu', kernel_initializer='random_normal', bias_initializer='zeros')(encoded)
encoded = Dense(90, activation = 'relu', kernel_initializer='random_normal', bias_initializer='zeros')(encoded)

decoded = Dense(120, activation = 'relu', kernel_initializer='random_normal', bias_initializer='zeros')(encoded)
decoded = Dense(150, activation = 'relu', kernel_initializer='random_normal', bias_initializer='zeros')(decoded)
decoded = Dense(784, activation='sigmoid', kernel_initializer='random_normal', bias_initializer='zeros')(decoded)

#Autoencoder: 
autoencoder = Model(input_img, decoded)

#Encoder:
encoder = Model(input=input_img, output=encoded)

#Decoder
encoded_input_1 = Input(shape=(90,))
encoded_input_2 = Input(shape=(120,))
encoded_input_3 = Input(shape=(150,))

decoder_layer_1 = autoencoder.layers[-3]
decoder_layer_2 = autoencoder.layers[-2]
decoder_layer_3 = autoencoder.layers[-1]

decoder_1 = Model(input = encoded_input_1, output = decoder_layer_1(encoded_input_1))
decoder_2 = Model(input = encoded_input_2, output = decoder_layer_2(encoded_input_2))
decoder_3 = Model(input = encoded_input_3, output = decoder_layer_3(encoded_input_3))

#Training in pre-training phase
sgd = optimizers.SGD(lr=0.1, momentum=0, decay=0, nesterov=False)
autoencoder.compile(optimizer=sgd, loss='mean_squared_error', metrics=['mae'])

train, train_targets = data_handling.read_train_dataset()
test, test_targets = data_handling.read_test_dataset()
train_targets = np_utils.to_categorical(train_targets)
test_targets = np_utils.to_categorical(test_targets)

hist = autoencoder.fit(train, train,
                epochs=40,
                batch_size=1, #default
                shuffle=True,
                validation_data=(test, test))

#-------------pretraining done, adding classifier-------------------

#adding one more "layer"
decoded = Dense(10, activation='sigmoid', kernel_initializer='random_normal', bias_initializer='zeros')(decoded)

classifier = Model(input = input_img, output = decoded)

#training classifier
sgd = optimizers.SGD(lr=0.1, momentum=0, decay=0, nesterov=False)
classifier.compile(optimizer=sgd, loss='mean_squared_error', metrics=['mae'])

classifier.fit(train, train_targets, 
                nb_epoch = 40, 
                batch_size = 1, 
                shuffle = True,
                validation_data=(test, test_targets))

classifier.predict(test)

encoded_imgs = encoder.predict(test)
decoded_imgs = decoder_1.predict(encoded_imgs)
decoded_imgs = decoder_2.predict(decoded_imgs)
decoded_imgs = decoder_3.predict(decoded_imgs)

#--------------------------------------------------------------------------------
# auto_imgs = autoencoder.predict(test)

n = 10  
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    # plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# np.save('3_hidden_layers_150_120_90_epoch50_lr01_batchsize1.npy', hist.history)
np.save('test_3_layers.npy', hist.history)

# plt.figure(2)
# plt.plot(hist.history['mean_absolute_error'])

plt.show()