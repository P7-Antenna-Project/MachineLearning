# --------------------------------------------------
# This file is used when models are already generated, and you want to test them on the data. If you want to use the same data as the models were trained on, use the same seed.
#       The file takes the models and calculates metrics and plots predictions. 
#       The models are loaded from the modelpath defined in the top, and the model you want to test is called by MODEL_FOR_TEST.
# --------------------------------------------------
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
#C:\Users\madsl\Desktop\DNN_results\Models
picklepath = "C:/Users/madsl/Dropbox/AAU/EIT 7. sem/P7/Python6_stuff/MachineLearning/data/"
modelpath = "C:/Users/madsl/Desktop/DNN_results/Models/"
save_fig_path = "C:/Users/madsl/Desktop/"
#"C:\Users\madsl\Dropbox\AAU\EIT 7. sem\P7\Python6_stuff\MachineLearning\data\COPYsimple_wire2_final_no_parametric.pkl"
# how many layers the model used for testing has (1-6)
MODEL_FOR_TEST = 4

# Seed for shuffling the data. The model is trained on 42.
SEED = 42

def load_data(path: str):
    """
    This function loads data from a pickle file located at the provided path.

    Parameters:
        path (str): The path to the pickle file.

    Returns:
        par_comb (np.ndarray): The parameter combinations.
        S11_par (np.ndarray): The best parametric data.
        frequency (np.ndarray): The frequency data.
        degrees (np.ndarray): The degrees data.
        combined_gain (np.ndarray): The combined gain list.
        std_dev (np.ndarray): The standard deviation of Phi.
        efficiency (np.ndarray): The efficiency data.
    """

    with open(path,'rb') as file:
        data_dict = pickle.load(file)
    print(f"Dictionary keys: {data_dict.keys()}")

    par_comb = np.asarray(data_dict['Parameter combination'])
    S11_vals = np.asarray(data_dict['S1,1'])
    frequency = np.asarray(data_dict['Frequency'])
    S11_parametrized = np.asarray(data_dict['Parametric S1,1'])
    degrees = np.asarray(data_dict['degrees'])
    combined_gain = np.asarray(data_dict['combined gain list'])
    std_dev = np.asarray(data_dict['Standard deviation Phi'])
    efficiency = np.asarray(data_dict['efficiency'])
    
    return par_comb, S11_vals, S11_parametrized, frequency, degrees, combined_gain, std_dev, efficiency

def normalize_data(data, inverse: bool):
    """
    This function normalizes data.

    Parameters: 
        data : The data to be normalized.
        inverse : If True, the data is de-normalized.
    Returns: 
        data_norm : The normalized data.
    """

    if inverse:
        data_norm = data*np.std(data) + np.mean(data)
    else:   
        mean = np.mean(data)
        std = np.std(data)
        data_norm = (data-mean)/std
    return data_norm

def plot_predictions(y_pred,y_test, MODEL_FOR_TEST):
    """
    This function plot predictions from a model.

    Parameters:
        y_pred : The predicted data.
        y_test : The test data.
        MODEL_FOR_TEST : The model used for testing. Load a model with load_model_calculate_metrics() and use the model number as input.


    Returns:
        A figure with 10 plots of the predicted and test data.
        Saves the figure in the save_fig_path defined in the top.
    """
    random_indices = random.sample(range(len(y_pred[1])), 10)
    plt.figure(figsize=(50, 50))
    for idx, i in enumerate(random_indices):
        ax = plt.subplot(5, 2, idx+1)
        ax.set_title(f"Model: {MODEL_FOR_TEST}, test sample: {i}", fontsize=30)
        ax.plot(frequency, y_test[i][:len(frequency)], label='test')
        ax.plot(frequency, y_pred[MODEL_FOR_TEST-1][i][:len(frequency)], label='pred')
        ax.legend()
        ax.grid(True)
        ax.set_ylim([-30,2])
    plt.savefig(f"{save_fig_path}test_pred_MODEL{MODEL_FOR_TEST}_{100}.png")    
    plt.close()

def load_model_calculate_metrics(model_layers, x_test, y_test, normalize_data):
    """
    This function loads models and calculates metrics.
    Parameters:
        model_layers : The models to be loaded.
        x_test : The test data.
        y_test : The test data.
        normalize_data : The function used to normalize the data.

    Returns:
        y_pred : The predicted data.
        y_pred_norm : The predicted data un-normalized (inverse norm. = true).
        mse : The mean squared error.
        mae : The mean absolute error.
    """

    for i in range (1,7):
        model_layers[i] = load_model(modelpath + f"MADSforward_model_{i}_layer.keras")
        print(f"Model {i} loaded")

    y_pred = [np.zeros(len(x_test)) for _ in range(7)]  # Changed from 6 to 7    
    y_pred_norm = [np.zeros(len(x_test)) for _ in range(7)]
    mse = np.zeros(7)
    mae = np.zeros(7)
    
    for i in range(1,7):
        y_pred[i] = model_layers[i].predict(x_test)
        y_pred_norm[i] = normalize_data(y_pred[i], inverse=True)
        mse[i] = np.mean((y_test - y_pred_norm[i])**2)
        mae[i] = np.mean(np.abs(y_test - y_pred_norm[i]))
        print(f"Model {i}:\nMSE: {mse[i]}\nMAE: {mae[i]}")
    
    return y_pred, y_pred_norm, mse, mae



if __name__ == "__main__":

    # Load the data
    par_comb, S11_par, frequency, degrees, combined_gain, std_dev, efficiency = load_data(f"{picklepath}COPYsimple_wire2_final_no_parametric.pkl")
    
    # Normalize the input data to the model
    par_comb_norm = normalize_data(par_comb, inverse=False)
    combined_gain_norm = normalize_data(combined_gain, inverse=False)
    std_dev_norm = normalize_data(std_dev, inverse=False)
    efficiency_norm = normalize_data(efficiency, inverse=False)
    S11_par_norm = normalize_data(S11_par, inverse=False)

    # Define input and output vectors
    input_vector = par_comb_norm
    output_vector = np.asarray([np.concatenate((S11_par_norm[i], [std_dev[i]], [efficiency_norm[i]]))for i in range(S11_par.shape[0])])
    
    # Define training and test data
    x_train, x_test, y_train, y_test = train_test_split(input_vector, output_vector, test_size=0.3, shuffle=True, random_state=SEED)
    
    model_layers = [None]*7
    
    # Load models and calculate metrics (MSE and MAE)
    y_pred, y_pred_norm, mse, mae = load_model_calculate_metrics(model_layers, x_test, y_test, normalize_data)

    # Plot predictions
    plot_predictions(y_pred, y_test, MODEL_FOR_TEST)