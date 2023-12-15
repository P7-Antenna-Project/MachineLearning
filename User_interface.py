import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pickle
from sklearn.model_selection import train_test_split
import keras.backend as K
import random
from keras.models import load_model
from MadsNeural_network_500epoch import *
import inquirer as inq
from pprint import pprint
import os
import argparse

# Create the parser
parser = argparse.ArgumentParser(description='User interface for antenna design')

# Add the arguments
parser.add_argument('-r', '--REAL_DATA', action='store_true', help='Use real data instead of simulated data')

# Parse the arguments
args = parser.parse_args()

# Set the REAL_DATA variable
REAL_DATA = args.REAL_DATA

if REAL_DATA:
    print("Running this shit fr fr")
# -------- Make user interface questions --------
questions = [
    inq.List('Antenna_type', message = "What type of antenna do you want to design ?", choices=['Wire', 'MIFA']),
    inq.Text('bandwidth', message="What is the desired bandwidth [MHz] ?"),
    inq.Text('center_frequency', message="What is the desired center frequency [MHz] ?")
    
]
answers = inq.prompt(questions)

# Print the user input
pprint(f' Design parameters: \n Bandwidth: {answers["bandwidth"]} MHz \n Center frequency: {answers["center_frequency"]} MHz')





# ------------ The following is necessary as the model is trained on data normalized wrt. the distribution of the whole dataset------------ 

if REAL_DATA:
    if answers['Antenna_type'] == 'Wire':
        file_path = "data\Wire_Results\WIRE_real_cst_BW_center.pkl"
    else:
        file_path = "data\MIFA_results\MIFA_real_cst_BW_center.pkl"

else:
    if answers['Antenna_type'] == 'Wire':
        file_path = "Data/WIRE_results/WIRE_Forward_results_with_band_centre.pkl"
    else:
        file_path = "Data/MIFA_results/MIFA_Forward_results_with_band_centre.pkl"

# Load data
with open(file_path, 'rb') as f:
    data_to_load = pickle.load(f)


bandwidth = data_to_load["bandwidth"]
center_frequency = data_to_load["centre_frequency"]
f1f2 = data_to_load["f1f2"]

parameter = data_to_load["Parameter combination"]

#Normalize the data wrt. distribution
parameter_norm = normalize_data(parameter,np.mean(parameter),np.std(parameter), False)
bandwidth_norm = normalize_data(bandwidth,np.mean(bandwidth),np.std(bandwidth), False)
center_frequency_norm = normalize_data(center_frequency,np.mean(center_frequency),np.std(center_frequency), False)



# Load the model
if answers['Antenna_type'] == 'Wire':
    model = load_model('data/Wire_Results/Reverse2ForwardWire_model.keras')
    
elif answers['Antenna_type'] == 'MIFA':
    model = load_model('data/MIFA_results/Reverse2ForwardMIFA_model.keras')
    

input_vector = np.asarray([[normalize_data(int(answers['bandwidth']), np.mean(bandwidth), np.std(bandwidth), False), normalize_data(int(answers['center_frequency']), np.mean(center_frequency), np.std(center_frequency), False)]])


prediction = normalize_data(model.predict(input_vector), np.mean(parameter), np.std(parameter), True)

if answers['Antenna_type'] == 'Wire':
    pprint(f'Predicted parameters: \n Length:  {prediction[0][0]} mm \n Height: {prediction[0][1]} mm \n Radius: {prediction[0][2]} mm \n')
elif answers['Antenna_type'] == 'MIFA':
    pprint(f'Predicted parameters: \n Tw1:  {prediction[0][0]} mm \n GroundingPinTopLength: {prediction[0][1]} mm \n Line1 height: {prediction[0][2]} mm \n Substrate height: {prediction[0][3]} mm')
