import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pickle
from sklearn.model_selection import train_test_split

# Flags
PLOT_TRAIN = False
PLOT_TEST = True
# Load the data
with open("data/Simple_wire.pkl", 'rb') as file:
    data_dict = pickle.load(file)

# Define the input and output data
x = np.asarray(data_dict['Parameter combination'])
frequency = np.asarray(data_dict['Frequency'])
y = np.asarray(data_dict['S1,1'])


# Convert y to linear scale
#y = 10**(y/10)

# Define training and test data
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
x_train = x[:int(0.7*x.shape[0]), :]
x_test = x[int(0.7*x.shape[0]):, :]

y_train = y[:int(0.7*y.shape[0]), :]
y_test = y[int(0.7*y.shape[0]):, :]

# # Construct a normalization layer for the input data
# norm_layer_data = keras.layers.Normalization()
# norm_layer_data.adapt(x_train)

# # Construct a normalization layer for the labels
# norm_layer_label = keras.layers.Normalization()
# norm_layer_label.adapt(y_train)

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
model.fit(
    x=x_train_norm,
    y=y_train_norm,
    batch_size=100,
    epochs=10000,
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
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss'])
    plt.subplot(122)
    plt.plot(mean_error)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['accuracy'])
    plt.show()

# Run the model on the test data
y_pred_norm = model.predict(x_test_norm)
 
# Reverse the normalization of the labels
y_pred = y_pred_norm*ystd1 + ymean1

# Plot the testing results
if PLOT_TEST:
    plt.figure()
    for i in range(5):
        plt.subplot(5, 1, i+1)
        plt.plot(frequency,y_pred[i])
        plt.plot(frequency,y_test[i])
        plt.legend(['pred', 'test'])
plt.show()
    