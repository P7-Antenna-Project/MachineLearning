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
from MadsNeural_network_500epoch import *
from Reduced_data_test import parse_bandwidth_center_freq

def parse_s11_files():
    with open("Reduced_data_Test\Wire_reduced_data_inverse2_pred.pkl", 'rb') as f:
        data_dict = pickle.load(f)
        print(data_dict.keys())
        parameter_comb = data_dict['Predictions']
        
    frequency_list = []   
    s11_list = []
    # Loop through the files
    for idx, i in tqdm(enumerate(range(0,len(parameter_comb)))):
        # print(f"RunID {i}") #Debugging
        filename = f"Reduced_data_Test/50_size_cst_curves_WIRE/S11{i}.txt"  
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
    return s11_list, frequency_list


if __name__ == "__main__":
    training_size = 0.5
    s11_list, frequency_list = parse_s11_files()
    
    bandwidth, center_freq, f1f2 = parse_bandwidth_center_freq(np.asarray(s11_list), np.asarray(frequency_list))
    
    print(bandwidth)
    with open("Reduced_data_Test/Wire_reduced_data_inverse2_pred.pkl", 'rb') as f:
        compare_dict = pickle.load(f)
    
        BW_CF = compare_dict['BW-CF']

    error =np.asarray([[BW_CF[i][0] - bandwidth[i],BW_CF[i][1]- center_freq[i]] for i in range(len(bandwidth))])
    plt.figure()
    plt.plot(np.asarray(BW_CF)[:31,1],bandwidth[:31],'o')
    plt.plot(np.asarray(BW_CF)[:31,1],np.asarray(BW_CF)[:31,0])
    plt.plot(np.asarray(BW_CF)[31:,1],bandwidth[31:])
    plt.plot(np.asarray(BW_CF)[31:,1],np.asarray(BW_CF)[31:,0])
    
    plt.figure()
    bandwidth_dif = np.asarray(bandwidth[:31])-np.asarray(bandwidth[31:])
    plt.plot(np.asarray(BW_CF)[:31,1],bandwidth_dif)
    plt.show()