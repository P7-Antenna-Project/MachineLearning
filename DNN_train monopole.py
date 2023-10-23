
########## Loading data from .mat file
########## Note: you can also use .csv or .txt, etc. files, it is easy to find code of loading them online.
import scipy.io
import numpy as np
import pandas as pd
x = np.loadtxt('MachineLearning/data/data112.csv', delimiter=',') 

# Load frequency range, vary the delimiter for even and odd lines in the file
    
# frequency_range = np.loadtxt(r"MachineLearning/data/s11/s11file_0.txt", delimiter='\t',skiprows=2)[:,0]


# # Define size of output array (s1,1 parameters)
# y = np.zeros(len(x),(np.max(frequency_range)-np.min(frequency_range))/(frequency_range[1]-frequency_range[0]))

# #Load all s11 parameters
# for i in range(len(x)):
#     y[i] = np.loadtxt(f'MachineLearning/data/s11/s11_{i}.txt', delimiter=',',skiprows=2)[:,1]



# im = y.split(f"\t")
# y[i][k][0] = float(im[0])
# y[i][k][1] = float(im[1])

# print(y.shape)
# x = scipy.io.loadmat('input.mat')
# y = scipy.io.loadmat('output.mat')


# Create an empty DataFrame to store the data
df = pd.DataFrame(columns=['S3,3/abs,dB'])
    
# Loop through the files
for i in range(0, len(x)):  
    filename = f"MachineLearning/data/s11/s11file_{i}.txt"    
    file_data = {'Frequency': [], 'S1,1': []}
    
    with open(filename, 'r') as file:
        file.readline()  # Skip the first line
        file.readline()  # Skip the second line
        for line in file:
            if line != '\n':  # Skip the empty line
                parts = line.split()
                file_data['Frequency'].append(float(parts[0]))
                file_data['S1,1'].append(float(parts[1]))
    
    # Create a DataFrame from the dictionary
    file_df = pd.DataFrame(file_data)
    # print(np.transpose(file_df['S1,1'].to_numpy()))
    # y[i,1]=np.transpose(file_df['S1,1'].to_numpy())
    
    

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