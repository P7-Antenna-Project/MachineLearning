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
PLOT_TEST_LOSS = True

# Load the data
with open("data\simple_wire2_final_with_parametric.pkl", 'rb') as file:
    data_dict = pickle.load(file)

def normalize_data(data_input, mean, std_dev, inverse: bool):
    if inverse:
        data = data_input*std_dev + mean
    else:   
        mean = mean
        std = std_dev
        data = (data_input-mean)/std
    return data

# Define the input and output data
y = np.asarray(data_dict['Parameter combination'])
frequency = np.asarray(data_dict['Frequency'])
x = np.asarray(data_dict['S1,1'])

# Define training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=42)

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
    layers.InputLayer(input_shape=(x.shape[1])),
    layers.Dense(256, activation='relu', name = 'layer1'),
    layers.Dense(256, activation='relu', name = 'layer2'),
    layers.Dense(256, activation='relu', name = 'layer3'),
    layers.Dense(256, activation='relu', name = 'layer4'),
#    layers.Dense(256, activation='relu', name = 'layer5'),
    layers.Dense(y_train_norm.shape[1], activation = 'linear', name = 'Output_layer')
])

model.summary()

# Compile the model with the solver adam and the loss function MSE
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.MeanAbsoluteError(),
    metrics=[keras.metrics.MeanSquaredError()]
)

#Create lists to store the results
loss_train = []
mean_error_train = []
mean_error_pred = np.zeros(50)
# Train the model
for j in range(1):
    model.fit(
        x=x_train_norm,
        y=y_train_norm,
        batch_size=100,
        epochs=500,
        shuffle=False,
        callbacks=[keras.callbacks.History()]
    )

    # Define the loss and accuracy for the training and test data
    loss_train.extend(model.history.history['loss'])
    mean_error_train.extend(model.history.history['mean_squared_error'])
    
    if PLOT_TRAIN:
        plt.figure()
        plt.subplot(121)
        plt.plot(np.array(loss_train).T)
        plt.ylabel('Absolute error')
        plt.xlabel('epoch')
        plt.legend(['Absolute error'])
        plt.ylim([0, 0.7])
        plt.subplot(122)
        plt.plot(10*np.log10(np.array(mean_error_train).T))
        plt.ylabel('Mean-squared error')
        plt.xlabel('epoch')
        plt.legend(['Mean-squared error'])
        #plt.ylim([0, 0.7])
        plt.savefig(f'data/DNN_results_reversal/train_loss/loss_{(j+1)*100}.png')
        plt.close()
    # Run the model on the test data and get the loss and mean-squared error
    y_pred_norm = model.predict(x_test_norm)
    _ , mean_error_pred[j] = model.evaluate(x_test_norm, y_test_norm)
    
    
    # Reverse the normalization of the labels
    y_pred = y_pred_norm*ystd1 + ymean1
    # Plot the testing results
    if PLOT_TEST:
        plt.figure(figsize=(50, 50))
        for i in range(10):
            plt.subplot(5, 2, i+1)
            plt.plot([1,2,3],y_pred[i], "r.")
            plt.plot([1,2,3],y_test[i], "b.")
            plt.legend(['pred', 'test'])
            #plt.ylim([0,15])
        plt.savefig(f'data/DNN_results_reversal/test_pred/test_pred_{j+1}k.png')
        plt.close()

# Plot the testing loss
if PLOT_TEST_LOSS:
    plt.plot(mean_error_pred)
    plt.ylabel('Mean-squared error')
    plt.xlabel('epoch')
    plt.savefig(f'data/DNN_results_reversal/test_loss/test_loss.png')
    plt.close()

model.save("Models/Reverse_Neural_network_3.keras")