import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import bz2
import scipy.optimize as opt
import random
try:
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
        print("Parsing s11 data")
        parameter_comb = np.loadtxt(par_comb_path, delimiter=',') 
    # Making list to store s11 values for each run
        s11_list = []
        frequency_list = []   
        # Loop through the files
        for idx, i in tqdm(enumerate(range(0,len(parameter_comb)))):
            # print(f"RunID {i}") #Debugging
            filename = f"{s11_path}"+ f"/s11_{i}.txt"  
            #print(filename) #Debugging
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
            #Debugging
            # print(f"index {idx}") 
            if idx > 0:
                if s11_list[idx] == s11_list[idx-1]:
                    error = 1/0
        print("Finished parsing s11 data")
        return s11_list, parameter_comb, frequency_list
except ZeroDivisionError:
    print("Error in s11 data")
    pass
    


def parse_gain(par_comb_path: str, PHI_gain_path, THETA_gain_path: str):
    """Parse the directivity files and return a list of gain/freq values for each run

    Args:
        par_comb_path (str): List of parameter combinations
        gain_path (str): File path to the gain files, expects the files to be named phi0_{i}.txt, theta45_{i}.txt etc.

    Returns:
        phi_gain_list (list): List of phi gain values for each run
        theta_gain_list (list): List of theta gain values for each run
        parameter_comb (list): List of parameter combinations
        degree_list (list): List of degree values 
    """    
    print("Parsing gain data")
    parameter_comb = np.loadtxt(par_comb_path, delimiter=',')
    phi_gain_list = []
    degree_list = []
    theta_gain_list = []
    std_dev_list = []
    max_gain_list = []
    # We skip 45 degrees since we dont have the counterpart at 135 degrees
    angles = [0,90]
    for i in tqdm(range(0, len(parameter_comb))):

        for j in angles:
            filename_phi = f"{PHI_gain_path}"+ f"/phi{j}_{i}.txt"
    
            with open(filename_phi, 'r') as file:
                file.readline()  # Skip the first line
                file.readline()  # Skip the second line
                #pass # add logic to read gain files

                for line in file:
                    if line != '\n':  # Skip the empty lines
                        parts = line.split()
                        phi_gain_list.append(float(parts[1]))
                        if i == 0: # We only need to store the frequency values once
                            degree_list.append(float(parts[0]))
        for j in angles:
            filename_theta = f"{THETA_gain_path}"+ f"/theta{j}_{i}.txt"
            with open(filename_theta, 'r') as file:
                file.readline()  # Skip the first line
                file.readline()  # Skip the second line
                for line in file:
                    if line != '\n':  # Skip the empty lines
                        parts = line.split()
                        theta_gain_list.append(float(parts[1]))
        combined_gain_list = phi_gain_list + theta_gain_list
        std_dev_list.append(np.std(combined_gain_list))
        max_gain_list.append(np.max(combined_gain_list))
                    
    dictionary = {'degrees': degree_list,
                  'phi_gain': phi_gain_list,
                  'theta_gain': theta_gain_list,
                  'Parameter combination': par_comb,
                  'Standard deviation': std_dev_list,
                  'Max gain': max_gain_list}
    print("Finished parsing gain data")
    return dictionary


def parameterize_s11(TEST_FLAG = False):
    print("Parameterizing s11 data")
    # Variables to tune the curve fitting
    max_poles = 10
    max_zeroes = 10
    pole_vals = [0, 400000]
    zero_vals = [0, 400000]

    def rational_func(x, *args):

        def rational(data, p, q):
            return np.polyval(q, data) / np.polyval(p + [1.0], data)

        # Split the args into poles and zeros
        mid_index = len(args) // 2
        poles = args[:mid_index]
        zeros = args[mid_index:]
        #print(poles, zeros)  # Debugging line
        return rational(x, [*poles], [*zeros])

    def fit_curve(data, frequency, poles, zeros):
        try:
            fitted_curve, cov_matrix, info_d, _, __ = opt.curve_fit(rational_func, frequency, data, p0=[*poles, *zeros], full_output=True)
            return fitted_curve, cov_matrix, info_d
        except RuntimeError:
            pass

    def find_best_curve(pole_vals, zero_vals,max_poles, max_zeroes, index):
        fvec = []
        pole_zero_comb = []
        
        parametric_data_list = []

        for i in range(1,max_poles):
            for j in range(1,max_zeroes):
                poles = np.linspace(pole_vals[0], pole_vals[1], i)
                zeroes = np.linspace(zero_vals[0], zero_vals[1], j)
            
                try:
                    parametric_data, _, info_d = fit_curve(s11[index], frequency, poles, zeroes)
                    parametric_data_list.append(parametric_data)
                    pole_zero_comb.append([i, j])
                    fvec.append(np.mean(info_d['fvec']**2))
                except TypeError:
                    pass
        best_comb = pole_zero_comb[np.argmin(fvec)]
        best_parametric_data = parametric_data_list[np.argmin(fvec)]
        return best_comb, best_parametric_data
    
    super_best_comb = []
    super_best_parametric_data = []
    if TEST_FLAG:
        # Loop through all runs and find the best curve
        random_indices = random.sample(range(len(s11)), 10)
        
        for run in tqdm(random_indices):
            best_comb, best_parametric_data  = find_best_curve(pole_vals, zero_vals, max_poles, max_zeroes, run)
            super_best_comb.append(best_comb)
            super_best_parametric_data.append(best_parametric_data)
        
        dictionary = {"Parameter combination": par_comb, "Frequency": frequency, "Best parametric data": super_best_parametric_data}

        plt.figure(figsize=(50, 50))
        for index, i in enumerate(random_indices):
            plt.subplot(5,2,index+1)
            plt.plot(frequency, s11[i], label="Original data")
            plt.plot(frequency, rational_func(frequency, *dictionary["Best parametric data"][index]), label="Parametric data")
            plt.legend()
        plt.show()
    else:
        # Loop through all runs and find the best curve
        for run in tqdm(range(len(s11))):
            best_comb, best_parametric_data  = find_best_curve(pole_vals, zero_vals, max_poles, max_zeroes, run)
            super_best_comb.append(best_comb)
            super_best_parametric_data.append(best_parametric_data)
        
        dictionary = {"Parameter combination": par_comb, "Frequency": frequency, "Best parametric data": super_best_parametric_data}
        print("Finished parameterizing s11 data")
    return dictionary
    
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
    

if __name__ == "__main__":
    s11, par_comb, frequency = parse_s11(par_comb_path = "data/par_comb_2508.csv", s11_path = "data/wireAntennaSimple2Results_nicolai/test_s11")
    
    # print(s11[0]==s11[1])
    # frequency = np.arange(500,3001, 2.5)
    # for index, value in enumerate(s11):
    #     plt.plot(frequency, value, label=f"RunID {index}")
    # plt.legend()
    # plt.show()
        
    
    # Add s11 variables to dictionary
    s11_dict = {'Parameter combination': par_comb, 'S1,1': np.asarray(s11), 'Frequency': frequency}
    
    gain_dict = parse_gain(par_comb_path = "data/par_comb_2508.csv", PHI_gain_path="data/wireAntennaSimple2Results_nicolai/test_phi", THETA_gain_path="data/wireAntennaSimple2Results_nicolai/test_theta")

    # Parameterize the s11 parameters for all runs
    s11_parameterized_dict = parameterize_s11(TEST_FLAG = False)
    
    #Combine the dictionaries
    combined_dict = s11_parameterized_dict | gain_dict
    combined_dict.update({'S1,1': np.asarray(s11)})
    # Save the data
    save_data(combined_dict, "data/Simple_wire_2_new_data.pkl")
    
    print("Finished parsing data")
