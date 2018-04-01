from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers
from keras.optimizers import SGD
import data_handling
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from keras.utils import np_utils
from sklearn.metrics import precision_recall_fscore_support

def evaluate(test_labels, predictions):
    precision, recall, f1score, support = precision_recall_fscore_support(test_labels, predictions)
    precision1=np.mean(precision)
    recall1=np.mean(recall)
    f1score1=np.mean(f1score)
    print("eval")
    # print('precision: {}'.format(precision))
    # print('recall: {}'.format(recall))
    # print('fscore: {}'.format(f1score))
    print('mean precision: {}'.format(precision1))
    print('mean recall: {}'.format(recall1))
    print('mean fscore: {}'.format(f1score1))


train, train_targets = data_handling.read_train_dataset()
test, test_targets = data_handling.read_test_dataset()
train_targets = np_utils.to_categorical(train_targets)
test_targets = np_utils.to_categorical(test_targets)

#-----------pre-training------------------------

# Layer 1
input_img = Input(shape = (784, ))
encoded1 = Dense(150, activation = 'sigmoid')(input_img)
decoded1 = Dense(784, activation = 'sigmoid')(encoded1)

autoencoder1 = Model(input = input_img, output = decoded1)
encoder1 = Model(input = input_img, output = encoded1)

# Layer 2
encoded1_input = Input(shape = (150,))
encoded2 = Dense(120, activation = 'sigmoid')(encoded1_input)
decoded2 = Dense(150, activation = 'sigmoid')(encoded2)

autoencoder2 = Model(input = encoded1_input, output = decoded2)
encoder2 = Model(input = encoded1_input, output = encoded2)

# Layer 3
encoded2_input = Input(shape = (120,))
encoded3 = Dense(90, activation = 'sigmoid')(encoded2_input)
decoded3 = Dense(120, activation = 'sigmoid')(encoded3)

autoencoder3 = Model(input = encoded2_input, output = decoded3)
encoder3 = Model(input = encoded2_input, output = encoded3)

# Deep Autoencoder
encoded1_da = Dense(150, activation = 'sigmoid')(input_img)
# encoded1_da_bn = BatchNormalization()(encoded1_da)
encoded2_da = Dense(120, activation = 'sigmoid')(encoded1_da)
# encoded2_da_bn = BatchNormalization()(encoded2_da)
encoded3_da = Dense(90, activation = 'sigmoid')(encoded2_da)
# encoded3_da_bn = BatchNormalization()(encoded3_da)
decoded3_da = Dense(120, activation = 'sigmoid')(encoded3_da)
decoded2_da = Dense(150, activation = 'sigmoid')(decoded3_da)
decoded1_da = Dense(784, activation = 'sigmoid')(decoded2_da)


sgd1 = SGD(lr = 1, decay = 0.5, momentum = .85)
sgd2 = SGD(lr = 1, decay = 0.5, momentum = .85)
sgd3 = SGD(lr = 1, decay = 0.5, momentum = .85)

autoencoder1.compile(loss='binary_crossentropy', optimizer = sgd1)
autoencoder2.compile(loss='binary_crossentropy', optimizer = sgd2)
autoencoder3.compile(loss='binary_crossentropy', optimizer = sgd3)

encoder1.compile(loss='binary_crossentropy', optimizer = sgd1)
encoder2.compile(loss='binary_crossentropy', optimizer = sgd1)
encoder3.compile(loss='binary_crossentropy', optimizer = sgd1)

deep_autoencoder = Model(input = input_img, output = decoded1_da)
deep_autoencoder.compile(loss='binary_crossentropy', optimizer = sgd1)

autoencoder1.fit(train, train,
                nb_epoch = 400, batch_size = 200,
                validation_split = 0.30,
                shuffle = True)

first_layer_code = encoder1.predict(train)

autoencoder2.fit(first_layer_code, first_layer_code,
                nb_epoch = 400, batch_size = 200,
                validation_split = 0.30,
                shuffle = True)

second_layer_code = encoder2.predict(first_layer_code)

autoencoder3.fit(second_layer_code, second_layer_code,
               nb_epoch = 400, batch_size = 200,
               validation_split = 0.30,
               shuffle = True)

third_layer_code = encoder3.predict(second_layer_code)


# Setting the weights of the deep autoencoder
deep_autoencoder.layers[1].set_weights(autoencoder1.layers[1].get_weights()) # first dense layer
# deep_autoencoder.layers[2].set_weights(autoencoder1.layers[3].get_weights()) # first bn layer
deep_autoencoder.layers[2].set_weights(autoencoder2.layers[1].get_weights()) # second dense layer
# deep_autoencoder.layers[4].set_weights(autoencoder2.layers[3].get_weights()) # second bn layer
deep_autoencoder.layers[3].set_weights(autoencoder3.layers[1].get_weights()) # thrird dense layer
# deep_autoencoder.layers[6].set_weights(autoencoder3.layers[3].get_weights()) # third bn layer
deep_autoencoder.layers[4].set_weights(autoencoder3.layers[2].get_weights()) # first decoder
deep_autoencoder.layers[5].set_weights(autoencoder2.layers[2].get_weights()) # second decoder
deep_autoencoder.layers[6].set_weights(autoencoder1.layers[2].get_weights()) # third decoder


#-------------pretraining done, adding classifier-------------------


# dense1 = Dense(784, activation = 'relu')(decoded1_da)
# dense1_drop = Dropout(.3)(dense1)
# #dense1_bn = BatchNormalization()(dense1_drop)
dense2 = Dense(10, activation = 'sigmoid')(decoded1_da)

classifier = Model(input = input_img, output = dense2)

sgd4 = SGD(lr = .1, decay = 0.001, momentum = .95, nesterov = True)
classifier.compile(loss='categorical_crossentropy', optimizer = sgd4, metrics=['accuracy'])
   
hist = classifier.fit(train, train_targets,
                nb_epoch = 400, batch_size = 1,
                validation_split = 0.3,
                shuffle = True)

decoded_imgs = deep_autoencoder.predict(test)
predictions = classifier.predict(test)
# predictions = np.argmax(val_preds, axis = 1)
# true_digits = np.argmax(test_targets, axis = 1)


evaluate(test_targets, np.array(predictions).astype(int)) #classifier or decdoded_imgs?
# evaluate(test_targets, np.array(predictions).astype(int))
# print(classifier.evaluate(predictions[0]))


#--------------------------------------------------------------------------------
#Plotting digits: 
n = 10  
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # np.save('test_3_layers_75epochs_0.3lr.npy', hist.history)
plt.show()

# Plotting images for layers: 

weights_3 = encoder_3.get_weights()[0]
weights_2 = encoder_2.get_weights()[0]
weights_1 = encoder_1.get_weights()[0]

# print(np.array(weights_3).shape)
# print(np.array(weights_2).shape)
# print(np.array(weights_1).shape)

plt.figure(figsize=(10, 10))
for i in range(150):
    plt.subplot(10, 15, i + 1)
    plt.imshow(weights_1[i].reshape((28, 28)), cmap=plt.cm.gray_r,
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
    plt.imshow(weights_3[i].reshape((10, 12)), cmap=plt.cm.gray_r,
                interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('90 components extracted by autoencoder layer 3', fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
plt.show()