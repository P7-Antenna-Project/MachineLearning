import numpy as np
import matplotlib.pyplot as plt
import pickle

with open("data/Simple_wire_2.pkl", 'rb') as file:
    data_dict = pickle.load(file)

# Define the input and output data
x = np.asarray(data_dict['Parameter combination'])
frequency = np.asarray(data_dict['Frequency'])
y = np.asarray(data_dict['S1,1'])


# Plot every 300th s11 curve
for i in range(0, len(x), 200):
    plt.figure()
    plt.plot(frequency, y[i])

plt.show()