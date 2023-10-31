import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pickle
from sklearn.model_selection import train_test_split

# Flags
PLOT_TRAIN = True
PLOT_TEST = True
# Load the data
with open("data/Simple_wire.pkl", 'rb') as file:
    data_dict = pickle.load(file)

# Define the input and output data
x = np.asarray(data_dict['Parameter combination'])
frequency = np.asarray(data_dict['Frequency'])
y = np.asarray(data_dict['S1,1'])

# Define training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=False)

# # Normalize training and test data
# x_train_norm = norm_layer_data(x_train)
# x_test_norm = norm_layer_data(x_test)
# y_train_norm = norm_layer_label(y_train)
# y_test_norm = norm_layer_label(y_test)

######### normalize input data
######### Note: you can also use other normalization method
xmean1 = np.mean(x_train)
xstd1 = np.std(x_train)
xmean2 = np.mean(x_test)
xstd2 = np.std(x_test)

x_train_norm = (x_train-xmean1)/xstd1
x_test_norm = (x_test-xmean1)/xstd1

########## normalize output data
######### Note: you can also use other normalization method
ymean1 = np.mean(y_train)
ystd1 = np.std(y_train)
ymean2 = np.mean(y_test)
ystd2 = np.std(y_test)

y_train_norm = (y_train-ymean1)/ystd1
y_test_norm = (y_test-ymean1)/ystd1

model = keras.Sequential([
    layers.InputLayer(input_shape=(3)),
    layers.Dense(64, activation='sigmoid', name = 'layer1'),
    layers.Dense(64, activation='sigmoid', name = 'layer2'),
    layers.Dense(64, activation='sigmoid', name = 'layer3'),
    layers.Dense(y_train_norm.shape[1], activation = 'linear', name = 'Output_layer')
])

model.summary()

# Compile the model with the solver adam and the loss function MSE
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.MeanAbsoluteError(),
    metrics=[keras.metrics.MeanSquaredError()]
)
    
# Train the model
for i in range(50):
    model.fit(
        x=x_train_norm,
        y=y_train_norm,
        batch_size=100,
        epochs=1000,
        shuffle=False,
        callbacks=[keras.callbacks.History()]
    )

    # Define the loss and accuracy for the training and test data
    loss = model.history.history['loss']
    mean_error = model.history.history['mean_squared_error']

    if PLOT_TRAIN:
        plt.figure()
        plt.subplot(121)
        plt.plot(loss)
        plt.ylabel('Absolute error')
        plt.xlabel('epoch')
        plt.legend(['Absolute error'])
        plt.subplot(122)
        plt.plot(mean_error)
        plt.ylabel('Mean-squared error')
        plt.xlabel('epoch')
        plt.legend(['Mean-squared error'])
        plt.savefig(f'data/DNN_results/loss_{i}k.png')
    # Run the model on the test data
    y_pred_norm = model.predict(x_test_norm)
    
    # Reverse the normalization of the labels
    y_pred = y_pred_norm*ystd1 + ymean1

    # Plot the testing results
    if PLOT_TEST:
        plt.figure(figsize=(50, 50))
        for i in range(50):
            plt.subplot(10, 5, i+1)
            plt.plot(frequency,y_pred[i])
            plt.plot(frequency,y_test[i])
            plt.legend(['pred', 'test'])
            plt.savefig(f'data/DNN_results/test_{i}k.png')
        