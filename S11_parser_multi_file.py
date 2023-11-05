import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle


def parse_s11(par_comb_path: str, s11_path: str):
    parameter_comb = np.loadtxt(par_comb_path, delimiter=',') 

    # Making list to store s11 values for each run
    s11_list = []
    frequency_list = []
    # Loop through the files
    for i in tqdm(range(0, len(parameter_comb))):  
        filename = f"{s11_path}"+ f"/s11file_{i}.txt"  
        
        with open(filename, 'r') as file:
            file.readline()  # Skip the first line
            file.readline()  # Skip the second line
            s11_values = [] # Create an empty array to store the s11 values

            # Loop through the lines in the file
            for line in file:
                if line != '\n':  # Skip the empty lines
                    parts = line.split()
                    s11_values.append(float(parts[1]))
                    if i == 0: # We only need to store the frequency values once
                        frequency_list.append(float(parts[0]))
                    
        s11_list.append(s11_values)
    return s11_list, parameter_comb, frequency_list

def parse_gain(par_comb_path: str, gain_path: str):
    parameter_comb = np.loadtxt(par_comb_path, delimiter=',')
    gain_list = []

    for i in tqdm(range(0, len(parameter_comb))):
        filename = f"{gain_path}"+ f"/gainfile_{i}.txt"
        with open(filename, 'r') as file:
            pass # add logic to read gain files
    return gain_list

def save_data(dictionary, path):
    # Pickle the dictionary into a file
    with open(path, 'wb') as pkl_path: 
        #Dump the pickle file at the given file path
        pickle.dump(dictionary, pkl_path)
    return



# print(f"The parameter combination is: {file_data['Parameter combination'][2]}, the s11 values are: {file_data['S1,1'][2]}")

if __name__ == "__main__":
    s11, par_comb, frequency = parse_s11(par_comb_path = "C:/Users/nlyho/Desktop/Simple_wire_2/data1320.csv", s11_path = "C:/Users/nlyho/Desktop/Simple_wire_2/Results")
    gain = parse_gain() #Work in progress

    #Add lists to dictionary
    file_data = {'Parameter combination': par_comb, 'S1,1': np.asarray(s11), 'Frequency': frequency}
    
    # Save the data
    save_data(file_data, "data/Simple_wire_2.pkl")