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

path = "C:/Users/madsl/Dropbox/AAU/EIT 7. sem/P7/Python6_stuff/MachineLearning/"

# Flags
PLOT_TRAIN = True
PLOT_TEST = True
PLOT_TEST_LOSS = True
MAX_LAYERS = 6

# Cleaning directories - remove old results and create new directories
def reset_dirs():
    current_time = time.strftime("%m%d-%H")  # get the current time
    for layer in range(MAX_LAYERS):
        #shutil.rmtree(f'{path}data/DNN_results/train_loss/MADStrain_loss_layer{layer+1}', ignore_errors=True)
        #shutil.rmtree(f'{path}data/DNN_results/test_pred/MADStest_pred_layer{layer+1}', ignore_errors=True)
        #shutil.rmtree(f'{path}data/DNN_results/test_loss/MADStest_loss_layer{layer+1}', ignore_errors=True)
        #shutil.rmtree(f'{path}data/DNN_results/Models', ignore_errors=True)

        os.mkdir(f'{path}data/DNN_results/sigmoid/train_loss/MADStrain_loss_layer{layer+1}_{current_time}')
        os.mkdir(f'{path}data/DNN_results/sigmoid/test_pred/MADStest_pred_layer{layer+1}_{current_time}')
        os.mkdir(f'{path}data/DNN_results/sigmoid/test_loss/MADStest_loss_layer{layer+1}_{current_time}')
        os.mkdir(f'{path}/data/DNN_results/sigmoid/Models_{current_time}')
    return

def load_data(path: str):
    """
    This function loads data from a pickle file located at the provided path.

    Parameters:
        path (str): The path to the pickle file.

    Returns:
        par_comb (np.ndarray): The parameter combinations.
        S11_vals(np.ndarray): The best parametric data.
        frequency (np.ndarray): The frequency data.
        degrees (np.ndarray): The degrees data.
        combined_gain (np.ndarray): The combined gain list.
        std_dev (np.ndarray): The standard deviation of Phi.
        efficiency (np.ndarray): The efficiency data.
    """

    with open(path,'rb') as file:
        data_dict = pickle.load(file)
    print(f"Dictionary keys: {data_dict.keys()}")

    par_comb = np.asarray(data_dict['Parameter combination'])
    S11_vals = np.asarray(data_dict['S1,1'])
    frequency = np.asarray(data_dict['Frequency'])
    S11_parametrized = np.asarray(data_dict['Parametric S1,1'])
    degrees = np.asarray(data_dict['degrees'])
    combined_gain = np.asarray(data_dict['combined gain list'])
    std_dev = np.asarray(data_dict['Standard deviation Phi'])
    efficiency = np.asarray(data_dict['efficiency'])
    
    return par_comb, S11_vals, S11_parametrized, frequency, degrees, combined_gain, std_dev, efficiency

def normalize_data(data, inverse: bool):
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
    par_comb, S11_vals, S11_parameterized, frequency, degrees, combined_gain, std_dev, efficiency = load_data(f"{path}data/simple_wire2_final_with_parametric.pkl")
    
    #Reshape parametrized S11 data, -1 means the size is inferred from the remaining dimensions
    s11_parameterized_flat = [np.reshape(arr, -1) for arr in S11_parameterized]
    
    # Normalize the input data to the model
    par_comb_norm = normalize_data(par_comb, inverse=False)
    combined_gain_norm = normalize_data(combined_gain, inverse=False)
    std_dev_norm = normalize_data(std_dev, inverse=False)
    efficiency_norm = normalize_data(efficiency, inverse=False)
    S11_vals_norm = normalize_data(S11_vals, inverse=False)
    s11_parameterized_flat_norm = normalize_data(s11_parameterized_flat, inverse=False)
    
    # Assuming combined_gain is a 2D array
    #combined_gain_norm_new = combined_gain_norm[:]

    print(f"par_comb_norm shape: {par_comb_norm.shape}")
    print(f"combined_gain_norm shape: {combined_gain_norm.shape}")
    print(f"std_dev_norm shape: {std_dev_norm.shape}")
    print(f"efficiency_norm shape: {efficiency_norm.shape}")
    print(f"S11_vals shape: {S11_vals.shape}")

    # Combine input data to a single vector
    #input_vector = np.hstack((par_comb_norm, combined_gain_norm_new, std_dev_norm))
    
    
    input_vector = par_comb_norm

    output_vector = np.asarray([np.concatenate((S11_vals_norm[i], s11_parameterized_flat_norm[i], [std_dev[i]], [efficiency_norm[i]]))for i in range(S11_vals.shape[0])])

    print(input_vector)
    print(f"input shape: {input_vector.shape}")
    print(output_vector.shape)
    print(output_vector)

    # Define training and test data
    x_train, x_test, y_train, y_test = train_test_split(input_vector, output_vector, test_size=0.3, shuffle=True, random_state=42)

    #normalize the data (maybe a keras normalization layer is better)
    # xmean1 = np.mean(x_train)
    # xstd1 = np.std(x_train)
    # xmean2 = np.mean(x_test)
    # xstd2 = np.std(x_test)

    # x_train_norm = normalize_data(x_train, inverse=False)
    # x_test_norm = normalize_data(x_test, inverse=False)

    # y_train_norm = normalize_data(y_train, inverse=False)
    # y_test_norm = normalize_data(y_test, inverse=False)

    # List containing the layers of the model, will have layers appended with each iteration
    base_layers = [
        layers.InputLayer(input_shape=input_vector.shape[1], name = 'Input_layer'),
        layers.Dense(y_train.shape[1], activation = 'linear', name = 'Output_layer')
    ]
    
    # Create list to store run times
    run_time = np.zeros(10)

    for layer in range(MAX_LAYERS):
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
                x=x_train,
                y=y_train,
                batch_size=100,
                epochs=400,
                shuffle=True,
                callbacks=[keras.callbacks.History()],
                verbose=1)
            print("j:",j)

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
                plt.savefig(f'{path}data/DNN_results/train_loss/MADStrain_loss_layer{layer+1}/loss_{(j+1)*100}.png')
                plt.close()

            # Run the model on the test data and get the loss and mean-squared error
            y_pred_norm = model.predict(x_test)
            _ , mean_error_pred[j] = model.evaluate(x_test, y_test)
            
            
            # Reverse the normalization of the labels
            y_pred = normalize_data(y_pred_norm, inverse=True)

            # Plot the testing results
            if PLOT_TEST:
                # Generate 10 unique random indices
                random_indices = random.sample(range(len(y_pred)), 10)

                plt.figure(figsize=(50, 50))
                for idx, i in enumerate(random_indices):
                    plt.subplot(5, 2, idx+1)
                    plt.plot(frequency, y_pred[i][0:1001], label='pred')
                    plt.plot(frequency, y_test[i][0:1001], label='test')
                    plt.legend()
                    plt.grid(True)
                    plt.ylim([-40,2])
                plt.savefig(f'{path}data/DNN_results/test_pred/MADStest_pred_layer{layer+1}/test_pred_{(j+1)*100}.png')
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
        model.save(f'{path}data/DNN_results/Models/MADSforward_model_{layer+1}_layer.keras', overwrite=True)
    plt.savefig(f'{path}data/DNN_results/test_loss/MADStest_loss_{layer+1}_layer.png')
    plt.close()
    np.savetxt(f'{path}data/DNN_results/MADSrun_time.txt', run_time)