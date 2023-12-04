from keras.models import load_model
import keras
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import curve_fit

def import_s11_data(filename):
    filename = filename  
            #print(filename) #Debugging
    with open(filename, 'r') as file:
        file.readline()  # Skip the first line
        file.readline()  # Skip the second line
        file.readline()  # Skip the third line
        s11_values = [] # Create an empty array to store the s11 values

        # Loop through the lines in the file
        for line in file:
            if line != '\n':  # Skip the empty lines
                parts = line.split()
                s11_values.append(float(parts[1]))

    return s11_values


def make_s11_curve(BW, center_freq, frequency):
    # Define frequency range
    frequency = frequency
    
    def rational_func(x, *args):

        def rational(data, q, p):
            return np.polyval(q, data) / np.polyval(p + [1.0], data)

        # Split the args into poles and zeros
        zeros = args[:len(args)//2]
        poles = args[len(args)//2:]
        #print(poles, zeros)  # Debugging line
        return rational(x, [*zeros], [*poles])

    # Define points for the S11 curve based on BW and center frequency
    points = []

    depth = -16
    a = (depth-(-10))/(BW/2)
    print(a)
    b1 = depth - a*center_freq
    b2 = depth + a*center_freq
    base_line = -3
    for i in frequency:

        def in_band(x):
            #print('in band called')
            return a*x + b1 if x < center_freq else -a*x + b2

        #print((-0.2-b2)/(-a))
        #print((-0.2-b1)/(a))
        points.append(in_band(i) if i < (base_line-b2)/(-a) and i > (base_line-b1)/(a) else base_line)
        
    

    #plt.plot(frequency, points, label='S11 box curve')
    zeros = np.linspace(int(10e3),int(10e6), 4)
    poles = np.linspace(int(10e3),int(10e6), 4)

    fitted_curve, cov_matrix, info_d, _, __ = curve_fit(rational_func, frequency, points, p0=[*zeros, *poles], full_output=True)

    S11_curve = rational_func(frequency, *fitted_curve)
    
    plt.plot(frequency, S11_curve, label='S11 fitted curve')
    return S11_curve

    

model_path = 'Models/Reverse_Neural_network_2.keras'
model = load_model(model_path)

#Load the data
with open("data/simple_wire2_final_with_parametric.pkl", 'rb') as file:
    data_dict = pickle.load(file)

# Define the input and output data
y = np.asarray(data_dict['Parameter combination'])
frequency = np.asarray(data_dict['Frequency'])
x = np.asarray(data_dict['S1,1'])

S_11_homemade = make_s11_curve(100, 1900, frequency)

# Define training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=42)

xmean1 = np.mean(x_train)
xstd1 = np.std(x_train)
xmean2 = np.mean(x_test)
xstd2 = np.std(x_test)

ymean2 = np.mean(y_test)
ystd2 = np.std(y_test)
ymean1 = np.mean(y_train)
ystd1 = np.std(y_train)


#S_11_for_test = np.asarray(import_s11_data(r'data\results_for_reverse_testingtxt.txt'))

#S_11_for_test[800:] = -3.3
#plt.plot(frequency, S_11_for_test, label='S11 test curve')

#S_11_for_test_norm = (S_11_for_test-xmean1)/xstd1

#print(f"Parameters of prediction {model.predict(np.asarray([S_11_for_test_norm]))*ystd1 + ymean1}")

x_train_norm = (x_train-xmean1)/xstd1
x_test_norm = (x_test-xmean1)/xstd1

S_11_homemade_norm = (S_11_homemade-xmean1)/xstd1

print(model.predict(np.asarray([S_11_homemade_norm]))*ystd1 + ymean1)

# Find test curves where the S11 goes below -10 dB and the frequency is below 2 GHz
test_indices = []
for idx, i in enumerate(x):
    if np.min(i) < -10 and frequency[np.argmin(i)] < 2000:
        test_indices.append(i)
print(f"Number of test curves that satisfy the condition: {len(test_indices)}")
random_indices = random.sample(test_indices, 3)

new_x_norm = (np.asarray(random_indices)-xmean1)/xstd1

print([model.predict(np.asarray([i]))*ystd1 + ymean1 for i in new_x_norm])

plt.legend()
plt.grid()
plt.xlabel('Frequency [MHz]')
plt.ylabel('S11 [dB]')
plt.show()

# for idx, i in enumerate(random_indices):
#     plt.plot(frequency, i, label=f'S11 {idx}')
# plt.legend()
# plt.show()