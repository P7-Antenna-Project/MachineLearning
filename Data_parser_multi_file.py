import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import bz2
import scipy.optimize as opt
import random

def parse_s11(par_comb_path: str, s11_path: str):
    """Parse the s11 files and return a list of s11 values for each run

    Args:
        par_comb_path (str): List of parameter combinations
        s11_path (str): File path to the s11 files, expects the files to be named s11file_0.txt, s11file_1.txt etc.

    Returns:
        s11_list (list): List of s11 values for each run
        parameter_comb (list): List of parameter combinations
        frequency_list (list): List of frequency values 
    """    
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

def rational(data, p, q):

    return np.polyval(p, data) / np.polyval(q + [1.0], data)

def rational3_3(x, *args):
    # Split the args into poles and zeros
    mid_index = len(args) // 2
    poles = args[:mid_index]
    zeros = args[mid_index:]
    return rational(x, [*poles], [*zeros])

def parameterize_data(data, frequency, poles, zeros):
    # print([*poles, *zeros])
    try:
        fitted_curve, cov_matrix, info_d, _, __ = opt.curve_fit(rational3_3, frequency, data, p0=[*poles, *zeros], full_output=True)
        return fitted_curve, cov_matrix, info_d
    except RuntimeError:
        pass
    
def save_data(dictionary: dict, path: str):
    """Save the data to a pickle file

    Args:
        dictionary (dict): Dictionary containing the data and labels
        path (str): File path to save the pickle file
    """    
    
    # Pickle the dictionary into a file
    with open(path, 'wb') as pkl_path: 
        #Dump the pickle file at the given file path
        pickle.dump(dictionary, pkl_path)
    return 


def find_best_curve(pole_vals, zero_vals,max_poles, max_zeroes, index):
    fvec = []
    pole_zero_comb = []
    
    parametric_data_list = []

    for i in range(1,max_poles):
        for j in range(1,max_zeroes):
            poles = np.random.randint(pole_vals[0], pole_vals[1], size=i)
            zeroes = np.random.randint(zero_vals[0], zero_vals[1], size=j)
            #poles = np.linspace(pole_vals[0], pole_vals[1], i)
            #zeroes = np.linspace(zero_vals[0], zero_vals[1], j)
            
            try:
                parametric_data, cov, info_d = parameterize_data(s11[index], frequency, poles, zeroes)
                parametric_data_list.append(parametric_data)
                pole_zero_comb.append([i, j])
                fvec.append(np.mean(info_d['fvec']**2))
                #print(f"amount of poles = {i}\n amount of zeros = {j} \n The condition number is: {np.linalg.cond(cov)}")
            except TypeError:
                pass
    # rational_curve_dict = {'Pole-zero combination': pole_zero_comb, 'MSE': MSE}
    best_comb = pole_zero_comb[np.argmin(fvec)]
    best_parametric_data = parametric_data_list[np.argmin(fvec)]
    
    return best_comb, best_parametric_data
# print(f"The parameter combination is: {file_data['Parameter combination'][2]}, the s11 values are: {file_data['S1,1'][2]}")

if __name__ == "__main__":
    s11, par_comb, frequency = parse_s11(par_comb_path = "C:/Users/nlyho/Desktop/Simple_wire_2/data1320.csv", s11_path = "C:/Users/nlyho/Desktop/Simple_wire_2/Results")
    #gain = parse_gain() #Work in progress

    #Add lists to dictionary
    # file_data = {'Parameter combination': par_comb, 'S1,1': np.asarray(s11), 'Frequency': frequency}

   
    
    # Parameterize the s11 parameters for all runs
    max_poles = 15
    max_zeroes = 15
    pole_vals = [1000, 3000]
    zero_vals = [1000, 3000]

    super_best_comb = []
    super_best_parametric_data = []
    random_indices = [1085, 1263, 19, 400, 302, 244, 769, 847, 208, 550]
    #random.sample(range(len(s11)), 10)
    print(random_indices)
    for run in tqdm(random_indices):
        best_comb, best_parametric_data  = find_best_curve(pole_vals, zero_vals, max_poles, max_zeroes, run)
        super_best_comb.append(best_comb)
        super_best_parametric_data.append(best_parametric_data)
    print(super_best_comb)
    # Add data to dictionary
    data_dict = {"Parameter combination": par_comb, "Frequency": frequency, "Best parametric data": super_best_parametric_data}
    
    # # Plot the best curve for a random run
    plt.figure(figsize=(50, 50))
    for index, i in enumerate(random_indices):
        plt.subplot(5,2,index+1)
        plt.plot(frequency, s11[i], label="Original data")
        plt.plot(frequency, rational3_3(frequency, *data_dict['Best parametric data'][index]), label="Parametric data")
        plt.legend()
    plt.show()
    
    
    # best_comb, best_parametric_data  = find_best_curve(pole_vals, zero_vals, max_poles, max_zeroes, 500)
    
    # max_poles = 10
    # max_zeroes = 10

    # pole_vals = [1000, 3000]
    # zero_vals = [1000, 3000]
    # fvec = []
    # pole_zero_comb = []
    
    # for i in range(1,max_poles):
    #     for j in range(1,max_zeroes):
    #         poles = np.random.randint(pole_vals[0], pole_vals[1], size=i)
    #         zeroes = np.random.randint(zero_vals[0], zero_vals[1], size=j)
    #         try:
    #             parametric_data, cov, info_d = parameterize_data(s11[1000], frequency, poles, zeroes)
    #             pole_zero_comb.append([i, j])
    #             fvec.append(np.mean(info_d['fvec']**2))
    #             print(f"amount of poles = {i}\n amount of zeros = {j} \n The condition number is: {np.linalg.cond(cov)}")
    #         except TypeError:
    #             pass


    
    
    # print(rational_curve_dict['Pole-zero combination'][0])
    # print(rational_curve_dict['Residuals'][0])
    # print(np.mean(np.asarray(fvec).T[1]**2))
    
    # Plot the parametric data for a single run
    # plt.plot(frequency, s11[500], label="Original data")

    # poles = np.linspace(pole_vals[0], pole_vals[1], best_comb[0])
    # zeroes = np.linspace(zero_vals[0], zero_vals[1], best_comb[1])

    # parametric_data, cov, info_d = parameterize_data(s11[1000], frequency, poles, zeroes)
    # plt.plot(frequency, rational3_3(frequency, *best_parametric_data), label="Parametric data")
    
    # Save the data
    #save_data(file_data, "data/Simple_wire_2.pkl")