import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import shutil
import random
import time
import json
import keras.backend as K
from keras import regularizers
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm


def load_data(path: str):
    """
    This function loads data from a pickle file located at the provided path.

    Parameters:
        path (str): The path to the pickle file.

    Returns:
        par_comb (np.ndarray): The parameter combinations.
        S11_par (np.ndarray): The best parametric data.
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
    # S11_parametrized = np.asarray(data_dict['Parametric S1,1'])
    degrees = np.asarray(data_dict['degrees'])
    combined_gain = np.asarray(data_dict['combined gain list'])
    std_dev = np.asarray(data_dict['Standard deviation Phi'])
    efficiency = np.asarray(data_dict['efficiency'])
    #efficiency = np.asarray(list(data_dict['efficiency'].values()))
    return par_comb, S11_vals, frequency, degrees, combined_gain, std_dev, efficiency


def normalize_data(data_input, mean, std_dev, inverse: bool):
    if inverse:
        data = data_input*std_dev + mean
    else:   
        mean = mean
        std = std_dev
        data = (data_input-mean)/std
    return data



# Define custom loss functions
def weighted_mse(y_true, y_pred):
    #Pass y_true values through a sigmoid function
    weights = 2* K.sigmoid(-y_true)+2

    return K.mean(weights * K.square(y_pred - y_true), axis=-1)


def weighted_loss_wire(y_true, y_pred):
    e = y_pred - y_true
    # Apply different weights based on the input parameters
    e1 = e[:, :2]*5 # Increase the weight for the first two parameters
    e2 = e[:, 2:]
    return K.mean(K.square(K.concatenate([e1, e2], axis=-1)), axis=-1)

def weighted_loss_MIFA(y_true, y_pred):
    e = y_pred - y_true
    # Apply different weights based on the input parameters
    e1 = e[:, 0]*5
    e2 = e[:, 2]*5# Increase the weight for the first two parameters
    e3 = e[:, 1]
    e4 = e[:, 3]
    
    return K.mean(K.square(K.concatenate([e1, e2, e3, e4], axis=-1)), axis=-1)

def train_forward_model(train_data_size):
    def weighted_mse(y_true, y_pred):
        #Pass y_true values through a sigmoid function
        weights = 2* K.sigmoid(-y_true)+2

        return K.mean(weights * K.square(y_pred - y_true), axis=-1)

    model = keras.Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(layers.Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.001)) )
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    # model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(y_train.shape[1], activation='linear'))

    # Create lists to store the results
    loss_train = []
    mean_error_train = []
    mean_error_pred = np.zeros(50)

    model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss=weighted_mse,
                metrics=[keras.metrics.MeanSquaredError()]
            )
    model.summary()


    # Train the model
    history = model.fit(x_train_norm,
                        y_train_norm,
                        epochs=500,
                        batch_size=100,
                        shuffle=True,
                        verbose=1)

    # Define the loss and accuracy for the training and test data
    loss_train.extend(model.history.history['loss'])
    mean_error_train.extend(model.history.history['mean_squared_error']) 
    # Run the model on the test data and get the loss and mean-squared error
    y_pred_norm = model.predict(x_test_norm)
    # _ , mean_error_pred = model.evaluate(X_test, y_test)
    y_pred = normalize_data(y_pred_norm, np.mean(y_test),np.std(y_test), inverse=True)


    # Find test curves where the S11 goes below -10 dB and the frequency is below 2 GHz
    test_indices = []
    for idx, i in enumerate(y_test[:,:1001]):
        if np.min(i) < -10 and frequency[np.argmin(i)] in range(1500,2500):
            test_indices.append(idx)
    print(f"Number of test curves that satisfy the condition: {len(test_indices)} within the test set")

    # Select 10 random curves from the good test curves
    random_indices = random.sample(test_indices, 10)


    plt.figure(figsize=(50, 50))
    for idx, i in enumerate(random_indices):
        plt.subplot(5, 2, idx+1)
        plt.plot(frequency, y_pred[i][:1001], label='pred')
        plt.plot(frequency, y_test[i][:1001], label='test')
        plt.legend()
        plt.grid(True)
        plt.ylim([-60,2])
    # plt.show()
    plt.savefig(f"Reduced_data_Test/Wire_reduced_data_pred_{training_size}.png")

    model.save(f'Reduced_data_Test/Wire_reduced_data_forward_model_{training_size}.keras', overwrite= True)
    return model

def interpolate_data():
    if WIRE_ANTENNA:
        # Generate random data for wire_length, wire_height and wire_radius within the minimum and maximum values
        wire_length =[157/10,157]
        wire_height = [4,15]
        wire_radius = [0.5,3]
        
        # Generate random data for the wire_length, wire_height and wire_width
        wire_length = np.random.uniform(wire_length[0],wire_length[1],100)
        wire_height = np.random.uniform(wire_height[0],wire_height[1],50)
        wire_radius = np.random.uniform(wire_radius[0],wire_radius[1],20)

        # Combine wire_length, wire_height and wire_width into a list of lists
        param_combination = [[w_l,w_h,w_r] for w_l in wire_length for w_h in wire_height for w_r in wire_radius]

        #Shuffle the columns in the list (does not move items between columns)
        random.seed(42)
        random.shuffle(param_combination)

    else:
        Tw1 = [2,8]
        groundingPinTopLength = [3,7]
        Line1_height = [8,16]
        substrateH = [0.7,0.9]
        
        Tw1 = np.random.uniform(Tw1[0],Tw1[1],30)
        groundingPinTopLength = np.random.uniform(groundingPinTopLength[0],groundingPinTopLength[1],30)
        Line1_height = np.random.uniform(Line1_height[0],Line1_height[1],20)
        substrateH = np.random.uniform(substrateH[0],substrateH[1],10)
        
        param_combination = [[Tw1_val,groundingPinTopLength_val,Line1_height_val,substrateH_val] for Tw1_val in Tw1 
                            for groundingPinTopLength_val in groundingPinTopLength 
                            for Line1_height_val in Line1_height 
                            for substrateH_val in substrateH]
        random.seed(42)
        random.shuffle(param_combination)
    return param_combination

def parse_bandwidth_center_freq(S11_curves, frequency):
    TEST_PLOT = True
    
    S11 = S11_curves
    freq = frequency
    print(S11.shape)

    # Look at each curve and find the bandwidth and centre frequency
    # Bandwidth is defined as the frequency range where the S11 is below -10dB
    # Centre frequency is defined as the frequency where the S11 is the lowest
    # The bandwidth and centre frequency is then saved in a list
    alpha = 10
    bandwidth = []
    centre_frequency = []
    f1f2 = []


    for idx, s11_val in (enumerate(tqdm(S11))):
        min_s11 = np.argmin(s11_val)
        # print(f"The lowest S11 value is {np.min(s11_val)} at index {min_s11}")
        # print(f'The frequency at the lowest S11 value is {freq[min_s11]}')
        centre_frequency.append(freq[min_s11])

        list_of_s11_below_10dB_index = []
        for idx, s11 in enumerate(s11_val):
            if s11 < -alpha:
                list_of_s11_below_10dB_index.append(idx)

        # print(f'The list of S11 below -10dB is {list_of_s11_below_10dB_index}')
        
        if len(list_of_s11_below_10dB_index) != 0:
            if len(np.arange(list_of_s11_below_10dB_index[0], list_of_s11_below_10dB_index[-1]+1)) == len(np.asarray(list_of_s11_below_10dB_index)):
                # print(f'The list of S11 below -10dB is single_band, the bandwidth is {freq[list_of_s11_below_10dB_index[-1]] - freq[list_of_s11_below_10dB_index[0]]}')
                bandwidth.append(freq[list_of_s11_below_10dB_index[-1]] - freq[list_of_s11_below_10dB_index[0]])
                f1f2.append([freq[list_of_s11_below_10dB_index[0]], freq[list_of_s11_below_10dB_index[-1]]])
            else:
                #By difference in index larger than 1, we can conclude that the bandwidth is not single band
                diff = np.diff(list_of_s11_below_10dB_index)
                im_list = []
                last_idx = 0
                for idx, k in enumerate(diff):
                    if k > 1:
                        im_list.append(list_of_s11_below_10dB_index[last_idx:idx+1])
                        last_idx = idx+1
                # Get the last bit with
                im_list.append(list_of_s11_below_10dB_index[last_idx:])

                for im in im_list:
                    if (np.asarray(im) == np.ones(len(im))*min_s11).any():
                        # print(freq[im[-1]] - freq[im[0]])
                        bandwidth.append(freq[im[-1]] - freq[im[0]])
                        f1f2.append([freq[im[0]], freq[im[-1]]])
        else:
            bandwidth.append(0)
            f1f2.append([0, 0])


    if TEST_PLOT:
        import matplotlib.pyplot as plt
        indexs = random.sample(range(0, len(S11)), 10)
        plt.figure(figsize=(50, 50))
        for idx, index in enumerate(indexs):
            plt.subplot(5, 2, idx+1)
            plt.grid(True)
            plt.plot(freq, S11[index], "r-")
            if bandwidth[index] != 0:
                plt.axvspan(f1f2[index][0], f1f2[index][1], alpha=0.5, color='blue')
            plt.hlines(-alpha,freq[0],freq[-1], linestyles='dashed')
        # plt.show()
        # plt.savefig(f"Reduced_data_Test/Wire_reduced_data_bandwidth_{training_size}.png")
        return bandwidth, centre_frequency, f1f2

def train_inverse_model(bandwidth_list,center_frequency_list,f1f2_list,parameter_comb):
    def weighted_loss(y_true, y_pred):
        e = y_pred - y_true
        # Apply different weights based on the input parameters
        e1 = e[:, :2]*5 # Increase the weight for the first two parameters
        e2 = e[:, 2:]
        return K.mean(K.square(K.concatenate([e1, e2], axis=-1)), axis=-1)

    
    bandwidth = bandwidth_list
    center_frequency = center_frequency_list
    f1f2 = f1f2_list

    parameter = parameter_comb

    #Normalize the data wrt. distribution
    parameter_norm = normalize_data(parameter,np.mean(parameter),np.std(parameter), False)
    bandwidth_norm = normalize_data(bandwidth,np.mean(bandwidth),np.std(bandwidth), False)
    center_frequency_norm = normalize_data(center_frequency,np.mean(center_frequency),np.std(center_frequency), False)

    input_vector =  np.asarray([[bandwidth_norm[x], center_frequency_norm[x]] for x in range(len(bandwidth))])
    input_vector2 = np.asarray([np.concatenate((input_vector[x], f1f2[x])) for x in range(len(input_vector))])
    output_vector = np.asarray(parameter_norm)

    print(f'Input vector shape: {input_vector.shape}')
    print(f'Output vector shape:{output_vector.shape}')

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(input_vector, output_vector, test_size=0.2, random_state=42)
    
        
    #Create the model
    model = keras.Sequential([
        layers.InputLayer(input_shape=(X_train.shape[1])),
        layers.Dense(256, activation='relu', name = 'layer1'),
        layers.Dense(256, activation='relu', name = 'layer2'),
        layers.Dense(256, activation='relu', name = 'layer3'),
        layers.Dense(256, activation='relu', name = 'layer4'),
        layers.Dense(256, activation='relu', name = 'layer5'),
        layers.Dense(y_train.shape[1], activation = 'linear', name = 'Output_layer')
    ])
    model.summary()
    # Compile model
    model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=[keras.metrics.MeanSquaredError()]
                )

    #Train model
    history = model.fit(X_train,
                        y_train,
                        epochs=500,
                        batch_size=200,
                        validation_split = 0.2)


        # Plot the testing results
    if PLOT_TEST:
        if WIRE_ANTENNA:
            # Make a grouped bar plot with the predicted parameters and the test parameters
            y_pred = model.predict(X_test)

            random_indices = random.sample(range(0, y_pred.shape[0]), 10)
            plt.figure(figsize=(10, 10))
            width = 0.1  # the width of the bars
            labels = ["Wire length", "Wire height", "Wire thickness"]
            for idx, i in enumerate(random_indices):
                plt.subplot(5, 2, idx+1)
                plt.grid(True)
                bars1 = plt.bar(np.arange(1,4) - width/2, normalize_data(y_pred[i],np.mean(parameter),np.std(parameter), True), width)
                bars2 = plt.bar(np.arange(1,4) + width/2, normalize_data(y_test[i],np.mean(parameter),np.std(parameter), True), width)
                plt.xticks(np.arange(1,4), labels)
                plt.legend([bars1, bars2], ['pred', 'test'])
            plt.savefig(f"Reduced_data_Test/Wire_reduced_data_inverse2_pred_{training_size}.png")

    # Save model
    if WIRE_ANTENNA:
        model.save(f'Reduced_data_Test/Wire_reduced_data_reverse2_{training_size}.keras')
    else:
        model.save('Reduced_data_Test/MIFA_reduced_data_reverse2.keras')

    return model
    
    
    
def load_forward_model():
    if WIRE_ANTENNA:
        model = keras.models.load_model('data/Wire_results/forward_model_wire.keras', custom_objects={'weighted_mse': weighted_mse})
    else:
        model = keras.models.load_model('data/MIFA_results/forward_model_MIFA.keras', custom_objects={'weighted_mse': weighted_mse})
    model.summary()
    return model

def load_inverse_model():
    if WIRE_ANTENNA:
        model = keras.models.load_model('data/wire_results/Wire_inverse_model_2.keras', custom_objects={'weighted_loss': weighted_loss_wire})
    else:
        model = keras.models.load_model('data/MIFA_results/MIFA_inverse_model_2.keras', custom_objects={'weighted_loss': weighted_loss_MIFA})
    model.summary()
    return model



if __name__ == '__main__':
    WIRE_ANTENNA = False
    training_size = 0.5    
    
    #Load data
    if WIRE_ANTENNA:
        par_comb, S11_vals, frequency, degrees, combined_gain, std_dev, efficiency = load_data("data/Wire_Results/simple_wire2_final_with_parametric.pkl")
    else:
        par_comb, S11_vals, frequency, degrees, combined_gain, std_dev, efficiency = load_data("data/MIFA_Results/MIFA_results.pkl")

    
    #---------Forward model-----------------
    print("Loading forward model")
    # Define input and output vectors for the forward model
    input_vector = par_comb

    output_vector = np.asarray([np.concatenate((S11_vals[i], [std_dev[i]], [efficiency[i]]))for i in range(S11_vals.shape[0])])

    training_size = training_size
    # Split data into training and test set
    x_train, x_test, y_train, y_test = train_test_split(input_vector, output_vector, test_size=(1-training_size),shuffle= True, random_state=42)

    # Normalize the data. X is the geometry parameters, y is the S11 values
    x_train_norm = normalize_data(x_train, np.mean(x_train),np.std(x_train), False)
    x_test_norm = normalize_data(x_test, np.mean(x_test),np.std(x_test), False)
    y_train_norm = normalize_data(y_train, np.mean(y_train),np.std(y_train), False)
    y_test_norm = normalize_data(y_test, np.mean(y_test),np.std(y_test), False)
    
    forward_model = load_forward_model()    
    
    
    #---------Data interpolation-----------------
    print("Interpolating data")
    interpolated_par_comb = interpolate_data()
    # Normalize the interpolated data with the same distribution as the training data in the forward model
    param_combination_norm = normalize_data(interpolated_par_comb, np.mean(x_train),np.std(x_train),False)

    # Predict the S11 values for the interpolated data
    forward_predictions_norm = forward_model.predict(param_combination_norm)
    
    # Reverse the normalization on the predicted S11 values
    
    interpolated_S11 = normalize_data(forward_predictions_norm[:,:1001], np.mean(y_train),np.std(y_train),True)
    

    #---------Bandwidth, center frequency parser-----------------
    # Parse the bandwidth and centre frequency for the interpolated data
    print("Parsing bandwidth and center frequency")
    bandwidth, center_frequency, f1f2 = parse_bandwidth_center_freq(interpolated_S11, frequency)
    
    
    #---------Inverse model 2-----------------
    print("Loading inverse model 2")
    #Define new input and output vectors for the inverse model
    input_vector =  np.asarray([[bandwidth[x], center_frequency[x]] for x in range(len(bandwidth))])
    input_vector2 = np.asarray([np.concatenate((input_vector[x], f1f2[x])) for x in range(len(input_vector))])
    output_vector = np.asarray(interpolated_par_comb)

    # Split data into training and testing
    x_train, x_test, y_train, y_test = train_test_split(input_vector, output_vector, test_size=0.3, random_state=42)

    #Normalize the data. X is bandwidth and center frequency, y is the geometry parameters
    x_train_norm = normalize_data(x_train,np.mean(x_train),np.std(x_train), False)
    x_test_norm = normalize_data(x_test,np.mean(x_test),np.std(x_test), False)
    y_train_norm = normalize_data(y_train,np.mean(y_train),np.std(y_train), False)
    y_test_norm = normalize_data(y_test,np.mean(y_test),np.std(y_test), False)

    # Load the inverse model
    inverse_model = load_inverse_model()
    
    #---- Define test goals for the inverse model ----
    print("Testing inverse model 2")
    # Make bandwidth and center frequency data for the inverse model to predict
    BW_test_vals = [50, 100]
    #Make center frequencies for the model to predict
    fc_test_vals = np.linspace(1000, 2500, 31, endpoint=True)
    
    
    # ---- Test the inverse model ----
    #Make input vector for the inverse model to predict
    test_input_vector = np.asarray([[bw,cf] for bw in BW_test_vals for cf in fc_test_vals])
    
    # Normalize the input vector
    test_input_vector_norm = normalize_data(test_input_vector, np.mean(x_train),np.std(x_train), False)
    
    # Predict the geometry parameters for the test input vector
    inverse_model_preds_norm = inverse_model.predict(test_input_vector_norm)
    
    # Reverse the normalization on the predicted geometry parameters
    inverse_model_preds = normalize_data(inverse_model_preds_norm, np.mean(y_train),np.std(y_train), True)

    
    
    # random_indices = random.sample(range(0, inverse_model_preds.shape[0]), 10)
    # plt.figure(figsize=(10, 10))
    # width = 0.1  # the width of the bars
    # labels = ["Wire length", "Wire height", "Wire thickness"]
    # for idx, i in enumerate(random_indices):
    #     plt.subplot(5, 2, idx+1)
    #     plt.grid(True)
    #     bars1 = plt.bar(np.arange(1,4) - width/2, inverse_model_preds, width)
    #     plt.xticks(np.arange(1,4), labels)
    #     plt.legend([bars1], ['pred'])

    # plt.show()
    print("Saving results")
    if WIRE_ANTENNA:
        with open("Reduced_data_Test/Wire_testing_inverse2_pred.pkl", 'wb') as f:
            pickle.dump({"Predictions" : inverse_model_preds, "BW-FC" : test_input_vector}, f)
    else:
        with open("Reduced_data_Test/MIFA_testing_inverse2_pred.pkl", 'wb') as f:
            pickle.dump({"Predictions" : inverse_model_preds, "BW-FC" : test_input_vector}, f)
    print("Finished program")



