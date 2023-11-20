import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pickle
from sklearn.model_selection import train_test_split
import os
import shutil
import random
import time

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

def load_data(path: str):
    with open(path,'rb') as file:
        data_dict = pickle.load(file)

    par_comb = np.asarray(data_dict['Parameter combination'])
    frequency = np.asarray(data_dict['Frequency'])
    S11_par = np.asarray(data_dict['Best parametric data'])
    return par_comb, S11_par, frequency

def normalize_data(data: dict, inverse: bool):
    if inverse:
        data_norm = data*np.std(data) + np.mean(data)
    else:   
        mean = np.mean(data)
        std = np.std(data)
        data_norm = (data-mean)/std
    return data_norm


# Run main code
if __name__ == "__main__":
        # Ready the directories
    reset_dirs()

    # Load the data
    x, y, frequency = load_data("data/Simple_wire_2_new_data.pkl")

    # Define training and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)

    #normalize the data (maybe a keras normalization layer is better)
    xmean1 = np.mean(x_train)
    xstd1 = np.std(x_train)
    xmean2 = np.mean(x_test)
    xstd2 = np.std(x_test)

    x_train_norm = normalize_data(x_train, inverse=False)
    x_test_norm = normalize_data(x_test, inverse=False)

    y_train_norm = normalize_data(y_train, inverse=False)
    y_test_norm = normalize_data(y_test, inverse=False)

    # List containing the layers of the model, will have layers appended with each iteration
    base_layers = [
        layers.InputLayer(input_shape=(3)),
        layers.Dense(y_train_norm.shape[1], activation = 'linear', name = 'Output_layer')
    ]
    
    # Create list to store run times
    run_time = np.zeros(10)

    for layer in range(10):
        start_time = time.perf_counter()
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
        
        # Create lists to store the results
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
                plt.subplots_adjust(wspace=0.5)
                plt.subplot(121)
                plt.plot(np.array(loss_train).T)
                plt.ylabel('Absolute error')
                plt.xlabel('epoch')
                plt.legend(['Absolute error'])
                plt.ylim([0, 1])
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
            y_pred = normalize_data(y_pred_norm, inverse=True)

            # Plot the testing results
            if PLOT_TEST:
                # Generate 10 unique random indices
                random_indices = random.sample(range(len(y_pred)), 10)

                plt.figure(figsize=(50, 50))
                for idx, i in enumerate(random_indices):
                    plt.subplot(5, 2, idx+1)
                    plt.plot(frequency, y_pred[i], label='pred')
                    plt.plot(frequency, y_test[i], label='test')
                    plt.legend()
                    plt.grid(True)
                    plt.ylim([-40,2])
                plt.savefig(f'data/DNN_results/test_pred/test_pred_layer{layer+1}/test_pred_{(j+1)*100}.png')
                plt.close()

        # Plot the testing loss
        if PLOT_TEST_LOSS:
            plt.plot(np.arange(1,51,1)*100, mean_error_pred, label = f'layer{layer+1}')
            plt.ylabel('Mean-squared error')
            plt.xlabel('epoch')
            plt.legend()
            plt.grid(True)
        # Save the run time for the current model
        run_time[layer] = time.perf_counter() - start_time
        # Save the model
        model.save(f'data/DNN_results/Models/forward_model_{layer+1}_layer.keras', overwrite=True)
    plt.savefig(f'data/DNN_results/test_loss/test_loss_{layer+1}_layer.png')
    plt.close()
    np.savetxt('data/DNN_results/run_time.txt', run_time)
    
