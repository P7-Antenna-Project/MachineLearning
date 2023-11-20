import numpy as np
import matplotlib.pyplot as plt
import pickle
import random

with open("data/Simple_wire_2_new_data.pkl", 'rb') as file:
    data_dict = pickle.load(file)

random_indices = random.sample(range(0, len(data_dict['S1,1'])), 10)

print(data_dict['S1,1'][628]==data_dict['S1,1'][629])
#plot random data
plt.figure(figsize=(50, 50))
for i in range(628, 629):
    plt.plot(data_dict['Frequency'], data_dict['S1,1'][i], label=f"Original data {i}")
plt.legend()
plt.show()