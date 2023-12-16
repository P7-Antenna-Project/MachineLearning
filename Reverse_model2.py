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
WIRE_ANTENNA = True

if WIRE_ANTENNA == True:
    file_path = "Data/WIRE_results/WIRE_Forward_results_with_band_centre.pkl"
else:
    file_path = "Data/MIFA_results/MIFA_Forward_results_with_band_centre.pkl"

# Load data
with open(file_path, 'rb') as f:
    data_to_load = pickle.load(f)

print(data_to_load.keys())

bandwidth = data_to_load["bandwidth"]
center_frequency = data_to_load["centre_frequency"]
f1f2 = data_to_load["f1f2"]

parameter = data_to_load["Parameter combination"]

# Extract data
print(data_to_load["bandwidth"])
input_vector =  np.asarray([[bandwidth[x], center_frequency[x]] for x in range(len(bandwidth))])
input_vector2 = np.asarray([np.concatenate((input_vector[x], f1f2[x])) for x in range(len(input_vector))])

output_vector = np.asarray(parameter)

print(input_vector.shape)
print(output_vector.shape)

# Split data into training and testing
x_train, x_test, y_train, y_test = train_test_split(input_vector, output_vector, test_size=0.2, random_state=42)

# Normalise data
x_train_mean = np.mean(x_train, axis=0)
x_train_std = np.std(x_train, axis=0)

y_train_mean = np.mean(y_train, axis=0)
y_train_std = np.std(y_train, axis=0)

x_train_norm = (x_train - x_train_mean)/x_train_std
y_train_norm = (y_train - y_train_mean)/y_train_std

x_test_norm = (x_test - x_train_mean)/x_train_std
y_test_norm = (y_test - y_train_mean)/y_train_std

# Create model
model = keras.Sequential([
    layers.InputLayer(input_shape=(x_train.shape[1])),
    layers.Dense(256, activation='relu', name = 'layer1'),
    layers.Dense(256, activation='relu', name = 'layer2'),
    layers.Dense(256, activation='relu', name = 'layer3'),
    layers.Dense(256, activation='relu', name = 'layer4'),
    layers.Dense(256, activation='relu', name = 'layer5'),
    layers.Dense(y_train.shape[1], activation = 'linear', name = 'Output_layer')
])

# Compile model
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=[keras.metrics.MeanSquaredError()]
              )

# Train model
history = model.fit(x_train, y_train, epochs=500, batch_size=100, validation_split = 0.2)

# Plot training history
plt.figure()
plt.plot(history.history['loss'])
plt.figure()
plt.plot(history.history['mean_squared_error'])
plt.show()

PLOT_TEST = True
# Plot the testing results
if PLOT_TEST:
    if WIRE_ANTENNA:
        # Make a grouped bar plot with the predicted parameters and the test parameters
        y_pred = model.predict(x_test_norm)

        random_indices = random.sample(range(0, y_pred.shape[0]), 10)
        plt.figure(figsize=(10, 10))
        width = 0.1  # the width of the bars
        labels = ["Wire length", "Wire height", "Wire thickness"]
        for idx, i in enumerate(random_indices):
            plt.subplot(5, 2, idx+1)
            plt.grid(True)
            bars1 = plt.bar(np.arange(1,4) - width/2, y_pred[i], width)
            bars2 = plt.bar(np.arange(1,4) + width/2, normalize_data(y_test[i],y_train_mean,y_train_std, True), width)
            plt.xticks(np.arange(1,4), labels)
            plt.legend([bars1, bars2], ['pred', 'test'])
        
    else:
        y_pred = model.predict(x_test_norm)
        random_indices = random.sample(range(0, y_pred.shape[0]), 10)
        plt.figure(figsize=(10, 10))
        width = 0.1  # the width of the bars
        labels = ['Tw1','groundingPinTopLength','Line1_height','substrateH']
        for idx, i in enumerate(random_indices):
            plt.subplot(5, 2, idx+1)
            plt.grid(True)
            bars1 = plt.bar(np.arange(1,5) - width/2, y_pred[i], width)
            bars2 = plt.bar(np.arange(1,4) + width/2, normalize_data(y_test[i],y_train_mean,y_train_std, True), width)
            plt.xticks(np.arange(1,5), labels)
            plt.legend([bars1, bars2], ['pred', 'test'])
    plt.show()




# Save model
if WIRE_ANTENNA == True:
    model.save('data/Wire_Results/Reverse2ForwardWire_model.keras')
else:
    model.save('models/MIFA_results/Reverse2ForwardMIFA_model.keras')