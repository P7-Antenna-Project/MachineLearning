from keras.models import load_model
import keras
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import curve_fit
from keras import regularizers
from MadsNeural_network_500epoch import normalize_data
# Load the data with parametric
with open('data/simple_wire2_final_with_parametric.pkl', 'rb') as f:
    parametric_data = pickle.load(f)

print(parametric_data.keys())
# Define parameter tensor
parameter_vector = np.asarray([[*zerosandpoles[0], *zerosandpoles[1]] for zerosandpoles in parametric_data["Parametric S1,1"]])
print(parameter_vector.shape)

# Define the geometric tensor
geometric_vector = np.asarray([geo for geo in parametric_data["Parameter combination"]])
print(geometric_vector.shape)



# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(parameter_vector, geometric_vector, test_size=0.2, random_state=42)

# Normalize the data
mean_x = X_train.mean()
std_x = X_train.std()
X_train_norm = (X_train - mean_x) / std_x
X_test_norm = (X_test - mean_x) / std_x

print(std_x, mean_x)

mean_y = np.mean(y_train)
std_y = np.std(y_train)
y_train_norm = (y_train - mean_y) / std_y
y_test_norm = (y_test - mean_y) / std_y
print(std_y, mean_y)
FORWARD = False

if FORWARD:
    # Make 5 layer nn with keras that has parameter_vector as input and geometric_vector as output
    model_forward = keras.Sequential()
    model_forward.add(keras.layers.Dense(100, input_shape=(geometric_vector.shape[1],), activation='relu'))
    model_forward.add(keras.layers.Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model_forward.add(keras.layers.Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model_forward.add(keras.layers.Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model_forward.add(keras.layers.Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model_forward.add(keras.layers.Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model_forward.add(keras.layers.Dense(parameter_vector.shape[1], activation='linear'))
    model_forward.compile(optimizer='adam', loss='mse', metrics=[keras.metrics.MeanSquaredError()])

    model_forward.summary()

    # Train the model
    history = model_forward.fit(y_train_norm, X_train_norm, epochs=250, batch_size=100, callbacks=[keras.callbacks.History()])
else:
    model_reverse = keras.Sequential()
    model_reverse.add(keras.layers.Dense(100, input_shape=(parameter_vector.shape[1],), activation='sigmoid'))
    model_reverse.add(keras.layers.Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model_reverse.add(keras.layers.Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model_reverse.add(keras.layers.Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model_reverse.add(keras.layers.Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model_reverse.add(keras.layers.Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model_reverse.add(keras.layers.Dense(geometric_vector.shape[1], activation='linear'))
    model_reverse.compile(optimizer='adam', loss='mse', metrics=[keras.metrics.MeanSquaredError()])

    model_reverse.summary()

    # Train the model
    history = model_reverse.fit(X_train_norm, y_train_norm, epochs=250, batch_size=100, callbacks=[keras.callbacks.History()])

PLOT_TEST = True
WIRE_ANTENNA = True

if PLOT_TEST:
    if WIRE_ANTENNA:
        if FORWARD:
            y_pred = model_forward.predict(y_test_norm)

            random_indices = random.sample(range(0, y_pred.shape[0]), 10)
            #plt.figure(figsize=(10, 10))
            width = 0.1  # the width of the bars
            labels = ["Wire length", "Wire height", "Wire thickness"]
            for idx, i in enumerate(random_indices): 
                print(f"The predicted value unormalized \n {normalize_data(y_pred[i],mean_x,std_x, True)} \n The target value \n {X_test[i]}")
                print(f"\n")
            losess = history.history.keys()
            print(losess)
            #plt.figure(figsize=(1, 2))
            fig, ax = plt.subplots(1, 2)
            plt.subplots_adjust(wspace=0.5)
            ax[0].set_title("Loss")
            ax[1].set_title("Mean Squared Error")
            ax[0].plot(history.history['loss'])
            ax[1].plot(history.history['mean_squared_error'])
            plt.savefig("forward_model_parametric_loss.png")
            np.save("forward_model_parametric_loss.npy", [history.history['loss'], history.history['mean_squared_error']])
            #plt.show()
                
        else:
            # Make a grouped bar plot with the predicted parameters and the test parameters
            y_pred = model_reverse.predict(X_test_norm)

            random_indices = random.sample(range(0, y_pred.shape[0]), 10)
            plt.figure(figsize=(10, 10))
            width = 0.1  # the width of the bars
            #labels = ["Wire length", "Wire height", "Wire thickness"]
            # for idx, i in enumerate(random_indices):
            #     plt.subplot(5, 2, idx+1)
            #     plt.grid(True)
            #     bars1 = plt.bar(np.arange(1,4) - width/2, normalize_data(y_pred[i],mean_y,std_y, True), width)
            #     bars2 = plt.bar(np.arange(1,4) + width/2, normalize_data(y_test_norm[i],mean_y,std_y, True), width)
            #     plt.xticks(np.arange(1,4), labels)
            #     plt.legend([bars1, bars2], ['pred', 'test'])
            #plt.savefig("inverse_model_parametric_predict.png")
            predictions = []
            targets = []
            for idx, i in enumerate(random_indices):
                predictions.append(normalize_data(y_pred[i],mean_y,std_y, True))
                targets.append(normalize_data(y_test_norm[i],mean_y,std_y, True))

            y_pred_real = normalize_data(y_pred, mean_y, std_y, True)
            y_test_real = normalize_data(y_test_norm, mean_y, std_y, True)
            mean_sqaured_error = np.mean(np.square(y_pred_real - y_test_real))

            print(f"Mean squared error: {mean_sqaured_error}")

            np.save("inverse_model_parametric_predict.npy", [predictions, targets])
            plt.figure(figsize=(1, 2))
            fig, ax = plt.subplots(1, 2)
            plt.subplots_adjust(wspace=0.5)
            ax[0].set_title("Loss")
            ax[1].set_title("Mean Squared Error")
            ax[0].plot(history.history['loss'])
            ax[1].plot(history.history['mean_squared_error'])
            plt.savefig("inverse_model_parametric_loss.png")
            np.save("inverse_model_parametric_loss.npy", [history.history['loss'], history.history['mean_squared_error']])
        
    else:
        y_pred = model.predict(X_test)
        random_indices = random.sample(range(0, y_pred.shape[0]), 10)
        plt.figure(figsize=(20, 20))
        width = 0.1  # the width of the bars
        labels = ['Tw1','groundingPinTopLength','Line1_height','substrateH']
        for idx, i in enumerate(random_indices):
            plt.subplot(5, 2, idx+1)
            plt.grid(True)
            bars1 = plt.bar(np.arange(1,5) - width/2, normalize_data(y_pred[i],np.mean(parameter),np.std(parameter), True), width)
            bars2 = plt.bar(np.arange(1,5) + width/2, normalize_data(y_test_norm[i], np.mean(parameter), np.std(parameter), True), width)
            plt.xticks(np.arange(1,5), labels)
            plt.legend([bars1, bars2], ['pred', 'test'])
    plt.show()