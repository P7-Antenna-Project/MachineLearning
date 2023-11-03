import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pickle
from sklearn.model_selection import train_test_split
import os
import shutil
from tqdm import tqdm

# Flags
PLOT_TRAIN = True
PLOT_TEST = True
PLOT_TEST_LOSS = True

def reset_dirs():
    for layer in range(10):
        shutil.rmtree(f'data/DNN_results/train_loss/train_loss_layer{layer+1}', ignore_errors=True)
        shutil.rmtree(f'data/DNN_results/test_pred/test_pred_layer{layer+1}', ignore_errors=True)
        
        os.mkdir(f'data/DNN_results/train_loss/train_loss_layer{layer+1}')
        os.mkdir(f'data/DNN_results/test_pred/test_pred_layer{layer+1}')
    return

def load_data(path):
    with open(path,'rb') as file:
        data_dict = pickle.load(file)
    x = np.asarray(data_dict['Parameter combination'])
    frequency = np.asarray(data_dict['Frequency'])
    y = np.asarray(data_dict['S1,1'])
    
    return x, y, frequency

# Ready the directories
reset_dirs()

# Load the data
with open("data/Simple_wire_2.pkl", 'rb') as file:
    data_dict = pickle.load(file)

# Define the input and output data
x = np.asarray(data_dict['Parameter combination'])
frequency = np.asarray(data_dict['Frequency'])
y = np.asarray(data_dict['S1,1'])

# Define training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)

# # Normalize training and test data
# x_train_norm = norm_layer_data(x_train)
# x_test_norm = norm_layer_data(x_test)
# y_train_norm = norm_layer_label(y_train)
# y_test_norm = norm_layer_label(y_test)

######### normalize input data
######### Note: you can also use other normalization method
xmean1 = np.mean(x_train)
xstd1 = np.std(x_train)
xmean2 = np.mean(x_test)
xstd2 = np.std(x_test)

x_train_norm = (x_train-xmean1)/xstd1
x_test_norm = (x_test-xmean1)/xstd1

########## normalize output data
######### Note: you can also use other normalization method
ymean1 = np.mean(y_train)
ystd1 = np.std(y_train)
ymean2 = np.mean(y_test)
ystd2 = np.std(y_test)

y_train_norm = (y_train-ymean1)/ystd1
y_test_norm = (y_test-ymean1)/ystd1

# List containing the layers of the model, will have layers appended with each iteration
base_layers = [
    layers.InputLayer(input_shape=(3)),
    layers.Dense(128, activation='sigmoid', name = f'layer1'),
    layers.Dense(128, activation='sigmoid', name = f'layer2'),
    layers.Dense(128, activation='sigmoid', name = f'layer3'),
    layers.Dense(y_train_norm.shape[1], activation = 'linear', name = 'Output_layer')
]



for layer in tqdm(range(3, 10, 1)):
    # Add a layer to the model
    base_layers.insert(-1, layers.Dense(128, activation='sigmoid', name = f'layer{layer+1}'))
    
    # Create the model
    model = keras.Sequential(base_layers,name=f'Sequential_{layer+1}')
    model.summary()
    # Compile the model with the solver adam and the loss function MSE
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.MeanAbsoluteError(),
        metrics=[keras.metrics.MeanSquaredError()]
    )
    
    #Create lists to store the results
    loss_train = []
    mean_error_train = []
    mean_error_pred = np.zeros(50)
    # Train the model
    for j in range(50):
        model.fit(
            x=x_train_norm,
            y=y_train_norm,
            batch_size=100,
            epochs=100,
            shuffle=True,
            callbacks=[keras.callbacks.History()],
            verbose=1)

        # Define the loss and accuracy for the training and test data
        loss_train.extend(model.history.history['loss'])
        mean_error_train.extend(model.history.history['mean_squared_error'])
        
        if PLOT_TRAIN:
            plt.figure()
            plt.subplot(121)
            plt.plot(np.array(loss_train).T)
            plt.ylabel('Absolute error')
            plt.xlabel('epoch')
            plt.legend(['Absolute error'])
            plt.ylim([0, 0.7])
            plt.subplot(122)
            plt.plot(np.array(mean_error_train).T)
            plt.ylabel('Mean-squared error')
            plt.xlabel('epoch')
            plt.legend(['Mean-squared error'])
            plt.ylim([0, 1])
            plt.savefig(f'data/DNN_results/train_loss/train_loss_layer{layer+1}/loss_{(j+1)*100}.png')
            plt.close()
        # Run the model on the test data and get the loss and mean-squared error
        y_pred_norm = model.predict(x_test_norm)
        _ , mean_error_pred[j] = model.evaluate(x_test_norm, y_test_norm)
        
        
        # Reverse the normalization of the labels
        y_pred = y_pred_norm*ystd1 + ymean1
        # Plot the testing results
        if PLOT_TEST:
            plt.figure(figsize=(50, 50))
            for i in range(0, len(y_pred), int(len(y_pred)/10)):
                plt.subplot(5, 2, int(i/(len(y_pred)/10)))
                plt.plot(frequency,y_pred[i], label='pred')
                plt.plot(frequency,y_test[i], label='test')
                plt.legend()
                plt.grid(True)
                plt.ylim([-40,0])
            plt.savefig(f'data/DNN_results/test_pred/test_pred_layer{layer+1}/test_pred_{(j+1)*100}.png')
            plt.close()

    # Plot the testing loss
    if PLOT_TEST_LOSS:
        plt.plot(np.arange(1,51,1)*100, mean_error_pred, label = f'layer{layer+1}')
        plt.ylabel('Mean-squared error')
        plt.xlabel('epoch')
plt.savefig(f'data/DNN_results/test_loss/test_loss_{layer+1}_layer.png')
plt.close()