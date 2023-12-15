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
import json
import keras.backend as K 
from keras import regularizers

path = "C:/Users/madsl/Dropbox/AAU/EIT 7. sem/P7/Python6_stuff/MachineLearning"
save_path = "C:/Users/madsl/Desktop"

# Flags
PLOT_TRAIN = True
PLOT_TEST = True
PLOT_TEST_LOSS = True
MAX_LAYERS = 6

# CHANGE IT IN LINE 165 ish AS WELL
activation_func = 'sigmoid'
plt.rcParams.update({'font.family': 'Serif','font.serif': 'CMU Serif', 'font.size': 14})


# Cleaning directories - remove old results and create new directories
def weighted_mse(y_true, y_pred):
#Pass y_true values through a sigmoid function
    weights = 2* K.sigmoid(-y_true)+2

    return K.mean(weights * K.square(y_pred - y_true), axis=-1)




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

def normalize_data(data_input, mean, std_dev, inverse: bool):
    if inverse:
        data = data_input*std_dev + mean
    else:   
        mean = mean
        std = std_dev
        data = (data_input-mean)/std
    return data

# Run main code
if __name__ == "__main__":
        # Ready the directories
    #reset_dirs()

    # Load the data
    par_comb, S11_vals, S11_parameterized, frequency, degrees, combined_gain, std_dev, efficiency = load_data(f"{path}/data/simple_wire2_final_with_parametric.pkl")
    
    #Reshape parametrized S11 data, -1 means the length of the array is inferred
    s11_parameterized_flat = [np.reshape(arr, -1) for arr in S11_parameterized]
    
 
 # Normalize data
    # par_comb_norm = normalize_data(par_comb,np.mean(par_comb),np.std(par_comb), False)
    # S11_vals_norm = normalize_data(S11_vals,np.mean(S11_vals),np.std(S11_vals), False)
    # # S11_parameterized_norm = normalize_data(S11_parameterized, np.mean(S11_parameterized),np.std(S11_parameterized), False)
    # frequency_norm = normalize_data(frequency, np.mean(frequency),np.std(frequency), False)
    # degrees_norm = normalize_data(degrees, np.mean(degrees),np.std(degrees), False)
    # combined_gain_norm = normalize_data(combined_gain, np.mean(combined_gain),np.std(combined_gain), False)
    # std_dev_norm = normalize_data(std_dev, np.mean(std_dev),np.std(std_dev), False)
    # efficiency_norm = normalize_data(efficiency, np.mean(efficiency),np.std(efficiency), False)
    # # Assuming combined_gain is a 2D array
    # #combined_gain_norm_new = combined_gain_norm[:]

    # print(f"par_comb_norm shape: {par_comb_norm.shape}")
    # print(f"combined_gain_norm shape: {combined_gain_norm.shape}")
    # print(f"std_dev_norm shape: {std_dev_norm.shape}")
    # print(f"efficiency_norm shape: {efficiency_norm.shape}")
    # print(f"S11_vals shape: {S11_vals.shape}")

    # Combine input data to a single vector
    #input_vector = np.hstack((par_comb_norm, combined_gain_norm_new, std_dev_norm))
    
    
    input_vector = par_comb

    output_vector = np.asarray([np.concatenate((S11_vals[i], [std_dev[i]], [efficiency[i]]))for i in range(S11_vals.shape[0])])

    print(input_vector)
    print(f"input shape: {input_vector.shape}")
    print(output_vector.shape)
    print(output_vector)

    # Define training and test data
    x_train, x_test, y_train, y_test = train_test_split(input_vector, output_vector, test_size=0.3, shuffle=True, random_state=42)
    std_dev = y_test[:,-2]
    efficiency = y_test[:,-1]
    
    x_train_norm = normalize_data(x_train, np.mean(x_train),np.std(x_train), False)
    x_test_norm = normalize_data(x_test, np.mean(x_test),np.std(x_test), False)
    y_train_norm = normalize_data(y_train, np.mean(y_train),np.std(y_train), False)
    y_test_norm = normalize_data(y_test, np.mean(y_test),np.std(y_test), False)


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
        # layers.Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.001)),
        # layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        # layers.Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.001)),
        # layers.Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.001)),
        # layers.Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.001)),
        # layers.Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.001)),
        layers.Dense(y_train.shape[1], activation = 'linear', name = 'Output_layer')
    ]
    
    # Create list to store run times
    run_time = np.zeros(10)

    for layer in range(MAX_LAYERS):
        start_time = time.perf_counter()
        # Add a layer to the model
        base_layers.insert(-1, layers.Dense(256, activation='sigmoid', name = f'layer{layer+1}',kernel_regularizer=regularizers.l2(0.001)))
        
        # Create the model
        model = keras.Sequential(base_layers,name=f'Sequential_{layer+1}')
        model.summary()

        # Compile the model with the solver adam and the loss function MSE
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=weighted_mse,
            metrics=[keras.metrics.MeanSquaredError()]
        )
        
        # Create lists to store the results
        loss_train = []
        mean_error_train = []
        mean_error_pred = np.zeros(10)
        # Train the model
        for j in range(10):
            model.fit(
                x=x_train_norm,
                y=y_train_norm,
                batch_size=100,
                epochs=100,
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
                plt.ylabel('Weighted Loss')
                plt.xlabel('epoch')
                plt.legend(['Weighted Loss'])
                plt.ylim([0, 3])
                plt.subplot(122)
                plt.plot(np.array(mean_error_train).T)
                plt.ylabel('Mean-squared error')
                plt.xlabel('epoch')
                plt.legend(['Mean-squared error'])
                plt.ylim([0, 1])
                # For saving the training loss figure
                train_loss_path = os.path.join(save_path,f'model_{layer+1}', f'loss_{(j+1)*100}.png').replace("\\", "/")
                plt.savefig(train_loss_path)
                plt.close()

            # Run the model on the test data and get the loss and mean-squared error
            y_pred_norm = model.predict(x_test_norm)
            _ , mean_error_pred[j] = model.evaluate(x_test_norm, y_test_norm)
            
            
            # Reverse the normalization of the labels

            y_pred = normalize_data(y_pred_norm, np.mean(y_train),np.std(y_train), inverse=True)
            # std_dev_pred = normalize_data(y_pred_norm[:,-2], np.mean(std_dev), np.std(std_dev), inverse=True)
            # efficiency_pred = normalize_data(y_pred_norm[:,-1], np.mean(efficiency), np.std(std_dev), inverse=True)

            # error_std_dev = np.abs(std_dev - std_dev_pred)
            # MSE_std_dev = np.mean(error_std_dev**2)
            
            # error_efficiency = np.abs(efficiency - efficiency_pred)
            # MSE_efficiency = np.mean(error_efficiency**2)

            # error_dictionary = {'error_std_dev': error_std_dev.tolist(), 'MSE_std_dev': MSE_std_dev.tolist(), 'error_efficiency': error_efficiency.tolist(), 'MSE_efficiency': MSE_efficiency.tolist()}
            # with open (f'{path}data/DNN_results/tanh/error_std_eff.txt', 'w') as file:
            #     file.write(json.dumps(error_dictionary))
                            
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
                # For saving the testing prediction figure
                test_pred_path = os.path.join(save_path,f'model_{layer+1}' , f'test_pred_{(j+1)*100}.png').replace("\\", "/")
                plt.savefig(test_pred_path)
                plt.close()

        # Plot the testing loss
        if PLOT_TEST_LOSS:
            plt.plot(np.linspace(1,10,10)*100, mean_error_pred, label = f'layer{layer+1}')
            plt.ylabel('Mean-squared error')
            plt.xlabel('epoch')
            plt.legend()
            plt.ylim([0, .4])
            plt.grid(True)
        # Save the run time for the current model
        run_time[layer] = time.perf_counter() - start_time
        # Save the model
        # model_path = os.path.join(path, 'data', 'DNN_results', 'tanh', 'Models', f'MADSforward_model_{layer+1}_layer.keras').replace("\\", "/")
        model.save(save_path + f'/MADSforward_model_{layer+1}_layer.keras', overwrite=True)
    # For saving the testing loss figure
    # test_loss_path = os.path.join(path, 'data', 'DNN_results', 'tanh', 'test_loss', f'MADStest_loss_{layer+1}_layer.png').replace("\\", "/")
    plt.savefig(save_path + f'/MADStest_loss_{layer+1}_layer.png')
    plt.close()


    # Save the run time
    # run_time_path = os.path.join(save_path, '/MADSrun_time.txt').replace("\\", "/")
    # np.savetxt(save_path + f'/run_time*')