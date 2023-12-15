import pickle
import numpy as np
import random
from tqdm import tqdm
WIRE_ANTENNA = False

TEST_PLOT = True


if WIRE_ANTENNA:
    with open("data/Wire_Results/forward_model_predictions_Wire.pkl", "rb") as f:
        data_to_be_loaded = pickle.load(f)
else:
    with open("data/MIFA_results/forward_model_predictions_MIFA.pkl", "rb") as f:
        data_to_be_loaded = pickle.load(f)

print(len(data_to_be_loaded))

S11 = data_to_be_loaded["S1,1"]
freq = data_to_be_loaded["Frequency"]
print(S11.shape)

# Look at each curve and find the bandwidth and centre frequency
# Bandwidth is defined as the frequency range where the S11 is below -10dB
# Centre frequency is defined as the frequency where the S11 is the lowest
# The bandwidth and centre frequency is then saved in a list
alpha = 10
bandwidth = []
centre_frequency = []
f1f2 = []


for idx, s11_val in tqdm(enumerate(S11)):
    min_s11 = np.argmin(s11_val)
    # print(f"The lowest S11 value is {np.min(s11_val)} at index {min_s11}")
    # print(f'The frequency at the lowest S11 value is {freq[min_s11]}')
    centre_frequency.append(freq[min_s11])

    list_of_s11_below_10dB_index = []
    for idx, s11 in enumerate(s11_val):
        if s11 < -alpha:
            list_of_s11_below_10dB_index.append(idx)

    # print(f'The list of S11 below -10dB is {list_of_s11_below_10dB_index}')
    
    if len(list_of_s11_below_10dB_index) != 0:
        if len(np.arange(list_of_s11_below_10dB_index[0], list_of_s11_below_10dB_index[-1]+1)) == len(np.asarray(list_of_s11_below_10dB_index)):
            # print(f'The list of S11 below -10dB is single_band, the bandwidth is {freq[list_of_s11_below_10dB_index[-1]] - freq[list_of_s11_below_10dB_index[0]]}')
            bandwidth.append(freq[list_of_s11_below_10dB_index[-1]] - freq[list_of_s11_below_10dB_index[0]])
            f1f2.append([freq[list_of_s11_below_10dB_index[0]], freq[list_of_s11_below_10dB_index[-1]]])
        else:
            #By difference in index larger than 1, we can conclude that the bandwidth is not single band
            diff = np.diff(list_of_s11_below_10dB_index)
            im_list = []
            last_idx = 0
            for idx, k in enumerate(diff):
                if k > 1:
                    im_list.append(list_of_s11_below_10dB_index[last_idx:idx+1])
                    last_idx = idx+1
            # Get the last bit with
            im_list.append(list_of_s11_below_10dB_index[last_idx:])

            for im in im_list:
                if (np.asarray(im) == np.ones(len(im))*min_s11).any():
                    # print(freq[im[-1]] - freq[im[0]])
                    bandwidth.append(freq[im[-1]] - freq[im[0]])
                    f1f2.append([freq[im[0]], freq[im[-1]]])
    else:
        bandwidth.append(0)
        f1f2.append([0, 0])


if TEST_PLOT:
    print(len(S11))
    print(len(centre_frequency))
    print(len(bandwidth))
    import matplotlib.pyplot as plt
    indexs =random.sample(range(0, len(S11)), 10)

    for idx, index in enumerate(indexs):
        plt.subplot(5, 2, idx+1)
        plt.grid(True)
        plt.plot(freq, S11[index], "r-")
        if bandwidth[index] != 0:
            plt.axvspan(f1f2[index][0], f1f2[index][1], alpha=0.5, color='blue')
        plt.hlines(-alpha,freq[0],freq[-1], linestyles='dashed')
    plt.show()


data_to_be_loaded["bandwidth"] = bandwidth
data_to_be_loaded["centre_frequency"] = centre_frequency
data_to_be_loaded["f1f2"] = f1f2

if WIRE_ANTENNA:
    with open("data/WIRE_results/WIRE_Forward_results_with_band_centre.pkl", "wb") as f:
        pickle.dump(data_to_be_loaded, f)
else:
    with open("data/MIFA_results/MIFA_Forward_results_with_band_centre.pkl", "wb") as f:
        pickle.dump(data_to_be_loaded, f)