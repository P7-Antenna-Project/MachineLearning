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
from Reduced_data_test import parse_bandwidth_center_freq

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




def parse_s11_files(WIRE_ANTENNA):
    if WIRE_ANTENNA:
        with open("Reduced_data_Test/Wire_testing_inverse2_pred.pkl", 'rb') as f:
            data_dict = pickle.load(f)
            print(data_dict.keys())
            parameter_comb = data_dict['Predictions']
    else:
        with open("Reduced_data_Test/MIFA_testing_inverse2_pred.pkl", 'rb') as f:
            data_dict = pickle.load(f)
            print(data_dict.keys())
            parameter_comb = data_dict['Predictions']
        
    frequency_list = []   
    s11_list = []
    # Loop through the files
    for idx, i in tqdm(enumerate(range(0,len(parameter_comb)))):
        # print(f"RunID {i}") #Debugging
        if WIRE_ANTENNA:
          filename = f"Reduced_data_Test/S11_Wire/S11_{i}.txt"  
        else:
            filename = f"Reduced_data_Test/S11_MIFA/S11_{i}.txt"
        #print(filename) #Debugging
        with open(filename, 'r') as file:
            file.readline()  # Skip the first line
            file.readline()  # Skip the second line
            s11_values = [] # Create an empty array to store the s11 values

            # Loop through the lines in the file
            for line in file:
                if line != '\n':  # Skip the empty lines
                    parts = line.split()
                    s11_values.append(float(parts[1]))
                    if i == 0: # We only need to store the frequency values once
                        frequency_list.append(float(parts[0]))
        
        s11_list.append(s11_values)
    return s11_list, frequency_list, data_dict


if __name__ == "__main__":
    WIRE_antennta = False
    s11_list, frequency_list, par_comb_dict = parse_s11_files(WIRE_ANTENNA=WIRE_antennta)
    
    
    bandwidth, center_freq, f1f2 = parse_bandwidth_center_freq(np.asarray(s11_list), np.asarray(frequency_list))
    
    # Save S11, frequency, BW and FC as npy files:
    if WIRE_antennta:
        np.save("Reduced_data_Test/saved_data/Wire_testing_inverse2_pred.npy", np.asarray(s11_list))
        np.save("Reduced_data_Test/saved_data/Wire_testing_inverse2_pred_freq.npy", np.asarray(frequency_list))
        np.save("Reduced_data_Test/saved_data/Wire_testing_inverse2_pred_BW.npy", np.asarray(bandwidth))
        np.save("Reduced_data_Test/saved_data/Wire_testing_inverse2_pred_FC.npy", np.asarray(center_freq))
    else:
        np.save("Reduced_data_Test/saved_data/MIFA_testing_inverse2_pred.npy", np.asarray(s11_list))
        np.save("Reduced_data_Test/saved_data/MIFA_testing_inverse2_pred_freq.npy", np.asarray(frequency_list))
        np.save("Reduced_data_Test/saved_data/MIFA_testing_inverse2_pred_BW.npy", np.asarray(bandwidth))
        np.save("Reduced_data_Test/saved_data/MIFA_testing_inverse2_pred_FC.npy", np.asarray(center_freq))
        
    print(bandwidth)
    with open("Reduced_data_Test/Wire_testing_inverse2_pred.pkl", 'rb') as f:
        compare_dict = pickle.load(f)
    
        BW_CF = compare_dict['BW-FC']

    error =np.asarray([[BW_CF[i][0] - bandwidth[i],BW_CF[i][1]- center_freq[i]] for i in range(len(bandwidth))])
    plt.figure()
    plt.plot(np.asarray(BW_CF)[:31,1],bandwidth[:31],'o')
    plt.plot(np.asarray(BW_CF)[:31,1],np.asarray(BW_CF)[:31,0])
    
    plt.figure()
    plt.plot(np.asarray(BW_CF)[31:,1],bandwidth[31:],'o')
    plt.plot(np.asarray(BW_CF)[31:,1],np.asarray(BW_CF)[31:,0])
    
    # plt.figure()
    # bandwidth_dif = np.asarray(bandwidth[:31])-np.asarray(bandwidth[31:])
    # plt.plot(np.asarray(BW_CF)[:31,1],bandwidth_dif)
    plt.show()