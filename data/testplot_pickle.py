import pickle
import matplotlib.pyplot as plt
import numpy as np


para_path = "C:/Users/madsl/Dropbox/AAU/EIT 7. sem/P7/Python6_stuff/MachineLearning/data" 
with open(f"{para_path}/Simple_wire_2_newdata_inc_eff.pkl", "rb") as f:
    dict = pickle.load(f)
print(dict.keys())

print(np.asarray(dict["combined gain list"]).shape)

# p i c k l e is split into:
# run ID | theta / phi | angle 0,45,90,135 | gain


run = 1500
print(dict["Standard deviation Phi"][run])

plt.figure(1)
for i in range(4):

    plt.polar(np.arange(0,73)*5*np.pi/180,dict["combined gain list"][run,0,i,:])
    plt.polar(np.arange(0,73)*5*np.pi/180,dict["combined gain list"][run,1,i,:])
plt.show()

