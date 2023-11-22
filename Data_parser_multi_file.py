import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import bz2
import scipy.optimize as opt
import random

# ----------------------------------------------
#              CODE FUNCTIONALITY:
# ----------------------------------------------
# This code parses the data from the simulation runs and saves it to a pickle file

# It CAN also parameterizes the s11 data and saves it to a pickle file,
#   but that takes a long time, so the parameterized data is already saved in the pickle file
#   comment some stuff if you want to parameterize the S11 again

# but! it saves the data in a picklefile called "data/simple_wire_2_new_data_madsTest_inc_eff.pkl", where the pickle contains :
#   dict_keys(['Parameter combination', 'S1,1', 'Frequency', 'degrees', 'combined gain list', 'Standard deviation Phi', 'efficiency'])


# REMEMBER TO SET DATA PATH IN LINE 267 ISH


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
    combined_gain_list = []
    #
    angles = [0,45,90,135]
    for i in tqdm(range(0, len(parameter_comb))):
        angle_list_theta = []
        angle_list_phi = []

        for j in angles:
            filename_phi = f"{PHI_gain_path}"+ f"/phi{j}_{i}.txt"
    
            with open(filename_phi, 'r') as file:
                file.readline()  # Skip the first line
                file.readline()  # Skip the second line
                #pass # add logic to read gain files
                phi_gain_list = []

                for line in file:
                    if line != '\n':  # Skip the empty lines
                        parts = line.split()
                        phi_gain_list.append(float(parts[1]))
                        if i == 0: # We only need to store the degree values once
                            degree_list.append(float(parts[0]))

            angle_list_phi.append(phi_gain_list)

        for j in angles:
            filename_theta = f"{THETA_gain_path}"+ f"/theta{j}_{i}.txt"
            with open(filename_theta, 'r') as file:
                file.readline()  # Skip the first line
                file.readline()  # Skip the second line

                theta_gain_list = []

                for line in file:
                    if line != '\n':  # Skip the empty lines
                        parts = line.split()
                        theta_gain_list.append(float(parts[1]))
            
            angle_list_theta.append(theta_gain_list)

        std_dev_list.append(np.std(np.ndarray.flatten(np.asarray(angle_list_phi)))) #)))))))))))
                        
        combined_gain_list.append([angle_list_phi, angle_list_theta])

        
        #max_gain_list.append(np.max(combined_gain_list))
                    
    dictionary = {'degrees': degree_list,
                  "combined gain list": np.asarray(combined_gain_list),
                  'Parameter combination': parameter_comb,
                  'Standard deviation Phi': std_dev_list}
                  #'Max gain': max_gain_list}
    print("Finished parsing gain data")
    return dictionary

#make dictionary for efficiency files:
def parse_efficiency(par_comb_path: str, efficiency_path: str):
    print("Parsing efficiency data")
    parameter_comb = np.loadtxt(par_comb_path, delimiter=',')
    efficiency_dict = {}
    for i in tqdm(range(0, len(parameter_comb))):
        filename = f"{efficiency_path}"+ f"/tot_eff_{i}.txt"
        with open(filename, 'r') as file:
            file.readline()  # Skip the first line
            file.readline()  # Skip the second line
            for line in file:
                if line != '\n':  # Skip the empty lines
                    parts = line.split()
                    efficiency_dict[i] = float(parts[1])
                    break  # Stop after reading the first non-empty line

    dictionary = {'efficiency': efficiency_dict,
                  'Parameter combination': parameter_comb}  # Use parameter_comb instead of par_comb
    
    print("Finished parsing efficiency data")
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
    # set data_path to correct:
    # ----------------------------------------------
    data_path = r"C:/Users/madsl/Dropbox/AAU/EIT 7. sem/P7/Python6_stuff/MachineLearning/data/wireAntennaSimple2Results_inc_eff"
    para_path = r"C:/Users/madsl/Dropbox/AAU/EIT 7. sem/P7/Python6_stuff/MachineLearning/data"
    # ----------------------------------------------

    #s11, par_comb, frequency = parse_s11(par_comb_path = f"{para_path}/par_comb_2508.csv", s11_path = f"{data_path}/test_s11")

    #phi, theta, par_comb, degree, standardDeviation  = parse_gain(par_comb_path = f"{para_path}/par_comb_2508.csv", PHI_gain_path=f"{data_path}/test_phi", THETA_gain_path=f"{data_path}/test_theta")

    #efficiency, par_comb = parse_efficiency(par_comb_path = f"{para_path}/par_comb_2508.csv", efficiency_path=f"{data_path}/test_eff")

    # print(s11[0]==s11[1])
    # frequency = np.arange(500,3001, 2.5)
    # for index, value in enumerate(s11):
    #     plt.plot(frequency, value, label=f"RunID {index}")
    # plt.legend()
    # plt.show()
        
    
    # Add s11 variables to dictionary
    #s11_dict = {'Parameter combination': par_comb, 'S1,1': np.asarray(s11), 'Frequency': frequency}
    
    gain_dict = parse_gain(par_comb_path = f"{para_path}/par_comb_2508.csv",  PHI_gain_path=f"{data_path}/test_phi", THETA_gain_path=f"{data_path}/test_theta")
    eff_dict = parse_efficiency(par_comb_path = f"{para_path}/par_comb_2508.csv", efficiency_path=f"{data_path}/test_eff")
    # Parameterize the s11 parameters for all runs
    #s11_parameterized_dict = parameterize_s11(TEST_FLAG = False)
    
    # open pickle Simple_wire_2.pkl and add eff_dict to it:
    with open(f"{para_path}/Simple_wire_2_new_data_madsTest_inc_eff.pkl", "rb") as f:
        previous_dict = pickle.load(f)
    previous_dict.update(gain_dict)
    previous_dict.update(eff_dict)
    save_data(previous_dict, f"{para_path}/Simple_wire_2_newdata_inc_eff.pkl")
    
    print(previous_dict.keys())


    #Combine the dictionaries
    #combined_dict = s11_parameterized_dict | gain_dict | eff_dict
    #combined_dict.update({'S1,1': np.asarray(s11)})
    # Save the data
    #save_data(combined_dict, "data/Simple_wire_2_new_data_madsTest_inc_eff.pkl")
    
    print("Finished parsing data")
   
