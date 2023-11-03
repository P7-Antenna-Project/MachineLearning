import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

x = np.loadtxt('C:/Users/nlyho/Desktop/Simple_wire_2/data1320.csv', delimiter=',') 

# Making list to store s11 values for each run
s11_list = []
frequency_list = []
# Loop through the files
for i in tqdm(range(0, len(x))):  
    filename = f"C:/Users/nlyho/Desktop/Simple_wire_2/Results/s11file_{i}.txt"    
    
    with open(filename, 'r') as file:
        file.readline()  # Skip the first line
        file.readline()  # Skip the second line
        s11_values = [] # Create an empty array to store the s11 values
         # Create list to store frequency values (The don't vary between runs)
        
        # Loop through the lines in the file
        for line in file:
            if line != '\n':  # Skip the empty line
                parts = line.split()
                s11_values.append(float(parts[1]))
                if i == 0: # We only need to store the frequency values once
                 frequency_list.append(float(parts[0]))
                 
    s11_list.append(s11_values)
    
# Add the parameter combinations and s11 values to the dictionary
file_data = {'Parameter combination': x, 'S1,1': np.asarray(s11_list), 'Frequency': frequency_list}

#Pickle the dictionary into a file
with open("C:/Users/nlyho/OneDrive - Aalborg Universitet/7. semester/Git/MachineLearning/data/Simple_wire_2.pkl", 'wb') as pkl_path: #Dump the pickle file at the given file path
    pickle.dump(file_data, pkl_path)


# print(f"The parameter combination is: {file_data['Parameter combination'][2]}, the s11 values are: {file_data['S1,1'][2]}")
