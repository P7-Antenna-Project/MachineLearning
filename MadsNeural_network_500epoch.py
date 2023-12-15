import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pickle
from sklearn.model_selection import train_test_split
import os
import shutil
import random
import time
from keras import backend as K
import json

# keep
path = "C:/Users/madsl/Dropbox/AAU/EIT 7. sem/P7/Python6_stuff/MachineLearning"
# path = "C:/Users/nlyho/OneDrive - Aalborg Universitet/7. semester/Git/MachineLearning"
print(path)

VERIFY_TEST_SET = False
PLOT_TRAIN = True
PLOT_TEST = True
PLOT_TEST_LOSS = True
MAX_LAYERS = 4

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
    #S11_parametrized = np.asarray(data_dict['Parametric S1,1'])
    degrees = np.asarray(data_dict['degrees'])
    combined_gain = np.asarray(data_dict['combined gain list'])
    std_dev = np.asarray(data_dict['Standard deviation Phi'])
    efficiency = np.asarray(data_dict['efficiency'])
    #efficiency = np.asarray(list(data_dict['efficiency'].values()))
    return par_comb, S11_vals, frequency, degrees, combined_gain, std_dev, efficiency

def normalize_data(data_input, mean, std_dev, inverse: bool):
    if inverse:
        data = data_input*std_dev + mean
    else:   
        mean = mean
        std = std_dev
        data = (data_input-mean)/std
    return data

def weighted_mse(y_true, y_pred):
#Pass y_true values through a sigmoid function
    weights = 2* K.sigmoid(-y_true)+2

    return K.mean(weights * K.square(y_pred - y_true), axis=-1)

# Run main code
if __name__ == "__main__":
    run_times = []
    start_time = time.perf_counter()
    #path = 'C:/Users/nlyho/OneDrive - Aalborg Universitet/7. semester/Git/MachineLearning/'
    path = "C:/Users/madsl/Dropbox/AAU/EIT 7. sem/P7/Python6_stuff/MachineLearning/"
    #par_comb, S11_vals, S11_parameterized, frequency, degrees, combined_gain, std_dev, efficiency = load_data(f"C:/Users/nlyho/OneDrive - Aalborg Universitet/7. semester/Git/MachineLearning/data/simple_wire2_final_with_parametric.pkl")
    par_comb, S11_vals, frequency, degrees, combined_gain, std_dev, efficiency = load_data(f"C:/Users/madsl/Dropbox/AAU/EIT 7. sem/P7/Python6_stuff/MachineLearning/data/MIFA_results/MIFA_results.pkl")

    # Normalize data
    par_comb_norm = normalize_data(par_comb,np.mean(par_comb),np.std(par_comb), False)
    S11_vals_norm = normalize_data(S11_vals,np.mean(S11_vals),np.std(S11_vals), False)
    # S11_parameterized_norm = normalize_data(S11_parameterized, np.mean(S11_parameterized),np.std(S11_parameterized), False)
    frequency_norm = normalize_data(frequency, np.mean(frequency),np.std(frequency), False)
    degrees_norm = normalize_data(degrees, np.mean(degrees),np.std(degrees), False)
    combined_gain_norm = normalize_data(combined_gain, np.mean(combined_gain),np.std(combined_gain), False)
    std_dev_norm = normalize_data(std_dev, np.mean(std_dev),np.std(std_dev), False)
    efficiency_norm = normalize_data(efficiency, np.mean(efficiency),np.std(efficiency), False)


    input_vector = par_comb_norm

    output_vector = np.asarray([np.concatenate((S11_vals_norm[i], [std_dev_norm[i]], [efficiency_norm[i]]))for i in range(S11_vals.shape[0])])


    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(input_vector, output_vector, test_size=0.3, shuffle=True, random_state=42)


    std_dev = y_test[:,-2]
    efficiency = y_test[:,-1]

    if VERIFY_TEST_SET:
        # Find test curves where the S11 goes below -10 dB and the frequency is below 2 GHz
        test_indices = []
        for idx, i in enumerate(normalize_data(y_test[:,:1001],np.mean(S11_vals),np.std(S11_vals), True)):
            if np.min(i) < -10 and frequency[np.argmin(i)] < 2000:
                test_indices.append(idx)
        print(f"Number of test curves that satisfy the condition: {len(test_indices)} within the test set")

        # Select 10 random curves from the good test curves
        random_indices = random.sample(test_indices, 10)
        random_indices = [321, 673, 43, 209, 495, 629, 868, 151, 755, 578]
        # plot random test data curves to verify that a good solution is within the test set
        plt.figure(figsize=(50, 50))
        for idx, i in enumerate(random_indices):
            plt.subplot(5, 2, idx+1)
            plt.plot(frequency, normalize_data(y_test[i][0:1001],np.mean(S11_vals),np.std(S11_vals),True), label='test')
            plt.legend()
            plt.grid(True)
            plt.ylim([-40,2])
        plt.show()
        
    # Make a 4 layer neural network using relu activation functions
    model = keras.Sequential()
    model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    # model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(y_train.shape[1], activation='linear'))

    # Create lists to store the results
    loss_train = []
    mean_error_train = []
    mean_error_pred = np.zeros(50)

    model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss=keras.losses.MeanAbsoluteError(),
                metrics=[keras.metrics.MeanSquaredError()]
            )
    model.summary()


    # Train the model
    history = model.fit(X_train,
                        y_train,
                        epochs=500,
                        batch_size=100,
                        shuffle=True,
                        verbose=1)

    # Define the loss and accuracy for the training and test data
    loss_train.extend(model.history.history['loss'])
    mean_error_train.extend(model.history.history['mean_squared_error']) 
            
    if PLOT_TRAIN:
        plt.figure()
        plt.subplots_adjust(wspace=0.5)
        plt.subplot(121)
        plt.plot(np.array(loss_train).T)
        plt.ylabel('Absolute error')
        plt.xlabel('epoch')
        plt.legend(['Absolute error'])
        plt.ylim([0, 1])
        plt.subplot(122)
        plt.plot(np.array(mean_error_train).T)
        plt.ylabel('Mean-squared error')
        plt.xlabel('epoch')
        plt.legend(['Mean-squared error'])
        plt.ylim([0, 1])
        
        # For saving the training loss figure
        train_loss_path = os.path.join(path, 'data', 'MIFA_results','train_loss.png').replace("\\", "/")
        #plt.show()
        plt.savefig(train_loss_path)
        plt.close()


        # Run the model on the test data and get the loss and mean-squared error
        y_pred_norm = model.predict(X_test)
        _ , mean_error_pred = model.evaluate(X_test, y_test)


        # Reverse the normalization of the data individually
        y_pred_s11 = normalize_data(y_pred_norm[:,:1001], np.mean(S11_vals),np.std(S11_vals), inverse=True)
        std_dev_pred = normalize_data(y_pred_norm[:,-2], np.mean(std_dev), np.std(std_dev), inverse=True)
        efficiency_pred = normalize_data(y_pred_norm[:,-1], np.mean(efficiency), np.std(std_dev), inverse=True)

        # Find test curves where the S11 goes below -10 dB and the frequency is below 2 GHz
        test_indices = []
        for idx, i in enumerate(y_pred_s11):
            if np.min(i) < -10 and frequency[np.argmin(i)] < 2000:
                test_indices.append(idx)
        print(f"Number of test curves that satisfy the condition: {len(test_indices)} within the test set")

        # Select 10 random curves from the good test curves
        random_indices = random.sample(test_indices, 10)
        random_indices = [321, 673, 43, 209, 495, 629, 868, 151, 755, 578]


        error_std_dev = np.abs(std_dev - std_dev_pred)
        MSE_std_dev = np.mean(error_std_dev**2)

        error_efficiency = np.abs(efficiency - efficiency_pred)
        MSE_efficiency = np.mean(error_efficiency**2)

        error_dictionary = {'error_std_dev': error_std_dev.tolist(), 'MSE_std_dev': MSE_std_dev.tolist(), 'error_efficiency': error_efficiency.tolist(), 'MSE_efficiency': MSE_efficiency.tolist()}
        print(f'MSE_std_dev: {error_dictionary["MSE_std_dev"]}, MSE_efficiency: {error_dictionary["MSE_efficiency"]}')

        
        # with open (f'{path}/data/DNN_results/relu/error_std_eff.txt', 'w') as file:
        #     file.write(json.dumps(error_dictionary)) 

    # Plot the test data and the predicted data
    if PLOT_TEST: 
        plt.figure(figsize=(50, 50))
        for idx, i in enumerate(random_indices):
            plt.subplot(5, 2, idx+1)
            plt.plot(frequency, y_pred_s11[i], label='pred')
            plt.plot(frequency, normalize_data(y_test[i][:1001],np.mean(S11_vals),np.std(S11_vals),True), label='test')
            plt.legend()
            plt.grid(True)
            plt.ylim([-40,2])
        #plt.show()
        # For saving the testing prediction figure
        test_pred_path = os.path.join(path, 'data', 'MIFA_results', f'test_pred.png').replace("\\", "/")
        plt.savefig(test_pred_path)
        plt.close()
            
        # Save the model(f"C:/Users/madsl/Dropbox/AAU/EIT 7. sem/P7/Python6_stuff/MachineLearning/data/MIFA_results/MIFA_results.pkl")
        modelpath = os.path.join(f"C:/Users/madsl/Dropbox/AAU/EIT 7. sem/P7/Python6_stuff/MachineLearning/data/MIFA_results/Test_forward_model1.keras")
        model.save(modelpath,overwrite=True)
            
        run_time = time.perf_counter() - start_time
        run_times.append(run_time)
        np.savetxt('run_times.txt', run_times)   
        
        if PLOT_TEST_LOSS:
            print(f'Test loss: {mean_error_pred}')