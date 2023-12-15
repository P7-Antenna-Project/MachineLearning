import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from MadsNeural_network_500epoch import load_data, weighted_mse, normalize_data
import pickle
import keras.backend as K

@keras.saving.register_keras_serializable()
def weighted_mse(y_true, y_pred):
#Pass y_true values through a sigmoid function
    weights = 2* K.sigmoid(-y_true)+2

    return K.mean(weights * K.square(y_pred - y_true), axis=-1)


# set path to desktop
path = 'C:/Users/madsl/Desktop/reLU/'
picklepath = "C:/Users/madsl/Dropbox/AAU/EIT 7. sem/P7/Python6_stuff/MachineLearning/data/simple_wire2_final_with_parametric.pkl"

par_comb, S11_vals, frequency, degrees, combined_gain, std_dev, efficiency = load_data(picklepath)



model = keras.models.load_model(path+'MADSforward_model_6_layer.keras')



input_vector = par_comb

output_vector = np.asarray([np.concatenate((S11_vals[i], [std_dev[i]], [efficiency[i]]))for i in range(S11_vals.shape[0])])

# print(input_vector)
# print(f"input shape: {input_vector.shape}")
# print(output_vector.shape)
# print(output_vector)

# Define training and test data
x_train, x_test, y_train, y_test = train_test_split(input_vector, output_vector, test_size=0.3, shuffle=True, random_state=42)
std_dev = y_test[:,-2]
efficiency = y_test[:,-1]
    
x_train_norm = normalize_data(x_train, np.mean(x_train),np.std(x_train), False)
x_test_norm = normalize_data(x_test, np.mean(x_test),np.std(x_test), False)
y_train_norm = normalize_data(y_train, np.mean(y_train),np.std(y_train), False)
y_test_norm = normalize_data(y_test, np.mean(y_test),np.std(y_test), False)



y_pred = normalize_data(model.predict(x_test_norm), np.mean(y_train), np.std(y_train), True)

plt.plot(frequency,y_test[1,:1001])
plt.plot(frequency,y_pred[1,:1001])
# plt.show()
percentage_error = []

MSE = [np.mean((y_test[run,:1001]-y_pred[run,:1001])**2) for run in range(y_pred.shape[0])]


[percentage_error.append(np.mean(np.abs((y_test[run,:1001]-y_pred[run,:1001])/y_test[run,:1001])*100)) for run in range(y_pred.shape[0])]

# print(percentage_error.shape) 
print(f'mean ={np.mean(MSE)})')
