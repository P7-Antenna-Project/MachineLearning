
########## Loading data from .mat file
########## Note: you can also use .csv or .txt, etc. files, it is easy to find code of loading them online.
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

# Load data from pickle file
with open("MachineLearning\data\Data_dict.pkl", 'rb') as file:
    data_dict = pickle.load(file)
    
#Define input and output data
x = data_dict['Parameter combination']
y = data_dict['S1,1']



# ########## Processing data
# import numpy as np
# from random import shuffle
# x = np.transpose(x)
# y = np.transpose(y)
# ######### transform from log10 to scalar
# ######### Note: you can also remove this step.
# y = 10**(y/10)
# ######### shuffle
# index = np.arange(x.shape[0])
# shuffle(index)
# x = x[index, :]
# y = y[index, :]
# ######### divide to train, test, and validation data = 7:1:2
# ######### Note: it is ok to divide into other ratio
# xtrain = x[:int(0.7*x.shape[0]), :]
# xtest = x[int(0.7*x.shape[0]):int(0.8*x.shape[0]), :]
# xval = x[int(0.8*x.shape[0]):, :]
# ytrain = y[:int(0.7*y.shape[0]), :]
# ytest = y[int(0.7*y.shape[0]):int(0.8*y.shape[0]), :]
# yval = y[int(0.8*y.shape[0]):, :]

# ######### normalize input data
# ######### Note: you can also use other normalization method
# xmean1 = np.mean(xtrain)
# xstd1 = np.std(xtrain)
# xmean2 = np.mean(xtest)
# xstd2 = np.std(xtest)
# xmean3 = np.mean(xval)
# xstd3 = np.std(xval)

# xtrain_norm = (xtrain-xmean1)/xstd1
# xtest_norm = (xtest-xmean1)/xstd1
# xval_norm = (xval-xmean1)/xstd1

# ########## normalize output data
# ######### Note: you can also use other normalization method
# ymean1 = np.mean(ytrain)
# ystd1 = np.std(ytrain)
# ymean2 = np.mean(ytest)
# ystd2 = np.std(ytest)
# ymean3 = np.mean(yval)
# ystd3 = np.std(yval)

# ytrain_norm = (ytrain-ymean1)/ystd1
# ytest_norm = (ytest-ymean1)/ystd1
# yval_norm = (yval-ymean1)/ystd1


# ######### build a model
# from tensorflow import keras

# model = keras.Sequential()
# model.add(keras.layers.InputLayer(input_shape=(xtrain_norm.shape[1],)))
# model.add(keras.layers.Dense(50, activation='relu'))
# model.add(keras.layers.Dense(50, activation='relu'))
# model.add(keras.layers.Dense(50, activation='relu'))
# model.add(keras.layers.Dense(ytrain_norm.shape[1], activation='linear'))

# ######### define loss function as "mse" and optimizer as "adam"
# model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mse', metrics=['accuracy'])

# ######### summarize the model information and print out
# model.summary()

# ######## train the model
# model.fit(
#     x=xtrain_norm,
#     y=ytrain_norm,
#     batch_size=100,
#     epochs=100,
#     validation_data=(xval_norm, yval_norm),
#     shuffle=True,
#     callbacks=[keras.callbacks.History()]
# )

# ######### retrieve results
# loss = model.history.history['loss']
# accuracy = model.history.history['accuracy']
# val_loss = model.history.history['val_loss']
# val_accuracy = model.history.history['val_accuracy']

# ######## plot training results
# import matplotlib.pyplot as plt

# plt.figure()
# plt.subplot(121)
# plt.plot(loss)
# plt.plot(val_loss)
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['loss', 'val_loss'])
# plt.subplot(122)
# plt.plot(accuracy)
# plt.plot(val_accuracy)
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['accuracy', 'val_accuracy'])
# plt.show()

# ######### test the model after training
# test_loss, test_accuracy = model.evaluate(xtest_norm, ytest_norm)
# ypred_norm = model.predict(xtest_norm)
# ypred = ypred_norm*ystd1+ymean1
# ypred = np.log10(ypred)*10
# ytest = np.log10(ytest)*10

# ######### plot test results
# plt.figure()
# for i in range(20):
#     plt.subplot(4, 5, i+1)
#     plt.plot(ypred[i])
#     plt.plot(ytest[i])
#     plt.legend(['pred', 'test'])
# plt.show()