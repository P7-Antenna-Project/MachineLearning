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



if answers['Antenna_type'] == 'Wire':
    def weighted_loss(y_true, y_pred):
        e = y_pred - y_true
        # Apply different weights based on the input parameters
        e1 = e[:, :2]*5 # Increase the weight for the first two parameters
        e2 = e[:, 2:]
        return K.mean(K.square(K.concatenate([e1, e2], axis=-1)), axis=-1)


else:
    def weighted_loss(y_true, y_pred):
        e = y_pred - y_true
        # Apply different weights based on the input parameters
        e1 = e[:, 0]*5
        e2 = e[:, 2]*5# Increase the weight for the first two parameters
        e3 = e[:, 1]
        e4 = e[:, 3]
        
        return K.mean(K.square(K.concatenate([e1, e2, e3, e4], axis=-1)), axis=-1)

# ------------ The following is necessary as the model is trained on data normalized wrt. the distribution of the whole dataset------------ 

if REAL_DATA:
    if answers['Antenna_type'] == 'Wire':
        file_path = "data\Wire_Results\WIRE_real_cst_BW_center.pkl"
    else:
        file_path = "data\MIFA_results\MIFA_real_cst_BW_center.pkl"

else:
    if answers['Antenna_type'] == 'Wire':
        file_path = "Data/WIRE_results/Wire_BW_fc.pkl"
    else:
        file_path = "Data/MIFA_results/MIFA_BW_fc.pkl"

# Load data
with open(file_path, 'rb') as f:
    data_to_load = pickle.load(f)


bandwidth = data_to_load["bandwidth"]
center_frequency = data_to_load["centre_frequency"]
f1f2 = data_to_load["f1f2"]

parameter = data_to_load["Parameter combination"]

input_vector =  np.asarray([[bandwidth[x], center_frequency[x]] for x in range(len(bandwidth))])
input_vector2 = np.asarray([np.concatenate((input_vector[x], f1f2[x])) for x in range(len(input_vector))])
output_vector = np.asarray(parameter)

# Split data into training and testing
x_train, x_test, y_train, y_test = train_test_split(input_vector, output_vector, test_size=0.3, random_state=42)

# #Normalize the data
# x_train_norm = normalize_data(x_train,np.mean(x_train),np.std(x_train), False)
# x_test_norm = normalize_data(x_test,np.mean(x_test),np.std(x_test), False)
# y_train_norm = normalize_data(y_train,np.mean(y_train),np.std(y_train), False)
# y_test_norm = normalize_data(y_test,np.mean(y_test),np.std(y_test), False)



# Load the model
if REAL_DATA:
    if answers['Antenna_type'] == 'Wire':
        model = load_model('data/Wire_Results/REAL_data_Reverse2ForwardWire_model.keras', custom_objects={'weighted_loss': weighted_loss})
    else:
        model = load_model('data/MIFA_results/REAL_data_Reverse2ForwardMIFA_model.keras',custom_objects={'weighted_loss': weighted_loss})
else:
    if answers['Antenna_type'] == 'Wire':
        model = load_model('data/Wire_Results/Wire_Inverse_model_2.keras',custom_objects={'weighted_loss': weighted_loss})
        
    elif answers['Antenna_type'] == 'MIFA':
        model = load_model('data/MIFA_results/MIFA_Inverse_model_2.keras',custom_objects={'weighted_loss': weighted_loss})
        

user_input_vector = np.asarray([[int(answers['bandwidth']), int(answers['center_frequency'])]])
user_input_norm = normalize_data(user_input_vector, np.mean(x_train), np.std(x_train), False)

prediction_norm = model.predict(user_input_norm)
prediction = normalize_data(prediction_norm, np.mean(y_train), np.std(y_train), True)

if answers['Antenna_type'] == 'Wire':
    pprint(f'Predicted parameters: \n Length:  {prediction[0][0]} mm \n Height: {prediction[0][1]} mm \n Radius: {prediction[0][2]} mm \n')
elif answers['Antenna_type'] == 'MIFA':
    pprint(f'Predicted parameters: \n Tw1:  {prediction[0][0]} mm \n GroundingPinTopLength: {prediction[0][1]} mm \n Line1 height: {prediction[0][2]} mm \n Substrate height: {prediction[0][3]} mm')
