from keras.models import load_model
import keras
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import matplotlib.pyplot as plt

model_path = 'Models/Reverse_Neural_network.keras'
model = load_model(model_path)

#Load the data
with open("data/Simple_wire.pkl", 'rb') as file:
    data_dict = pickle.load(file)

# Define the input and output data
y = np.asarray(data_dict['Parameter combination'])
frequency = np.asarray(data_dict['Frequency'])
x = np.asarray(data_dict['S1,1'])

# Define training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=False)

xmean1 = np.mean(x_train)
xstd1 = np.std(x_train)
xmean2 = np.mean(x_test)
xstd2 = np.std(x_test)

ymean2 = np.mean(y_test)
ystd2 = np.std(y_test)
ymean1 = np.mean(y_train)
ystd1 = np.std(y_train)


x_train_norm = (x_train-xmean1)/xstd1
x_test_norm = (x_test-xmean1)/xstd1

new_x = []
for i in frequency:
    new_x.append(-10 if i < 1950 and i > 1850 else 0)

new_x_norm = (new_x-xmean1)/xstd1

print(model.predict(np.asarray([new_x_norm]))*ymean1 + ystd1)

plt.plot(frequency, new_x, label = 'Input')
plt.show()