import numpy as np
import pandas as pd
import sys
from tqdm import tqdm

# -------------------------------------- path til workspace: --------------------------------------
#mads path:
#path = "C:/Users/madsl/Dropbox/AAU/EIT 7. sem/P7/Python6_stuff/MachineLearning"
#result_path = "/data/wireA<ntennaSimple2Results_inc_eff"
# Nicolai path: 
#path = "C:/Users/nlyho/OneDrive - Aalborg Universitet/7. semester/Git/MachineLearning"
# Anders path:
path = "C:/Users/bundg/OneDrive/Dokumenter/Python_36/MachineLearning"

# HUSK AT SÆTTE START OG STOP ( LINJE 57+59 )

laptop_path = r"C:\Program Files (x86)\CST Studio Suite 2023\AMD64\python_cst_libraries"
sys.path.append(laptop_path)
import cst.interface
import shutil

#new_cstfile = r"C:\Users\nlyho\OneDrive - Aalborg Universitet\7. semester\Git\MachineLearning\CST files\Wire_antenna_simple_2.cst"
new_cstfile = f"{path}\CST files\Build_voxel_model_0_5.cst"
#new_cstfile = f"{path}\CST files\Build_voxel_model.cst"


DE = cst.interface.DesignEnvironment()
microwavestructure = DE.open_project(new_cstfile)
modeler = microwavestructure.modeler
schematic = microwavestructure.schematic


# generate the complete vba code
def runmacro(voxel_pos,batch): #commented s21file 
    #print(voxel_pos)
    super_code = []
    super_code.append('Option Explicit')
    super_code.append('Sub Main')
    super_code.append('StartVersionStringOverrideMode "2023.5|32.0.1|20230608"')
    super_code.append('Component.New "component{}"'.format(batch+1))
    #print(voxel_pos)
    for i in range(number_of_solids_created_at_a_time):
        code = [
            # Merged Block - define brick: component1:solid1
            'With Brick',
                '.Reset' ,
                '.Name "solid{}_{}"'.format(batch+1,i) ,
                '.Component "component{}"'.format(batch+1) , 
                '.Material "PEC"',
                '.Xrange "{}", "{}"'.format(np.round(voxel_pos[i,0]-voxel_size/2,2), np.round(voxel_pos[i,0]+voxel_size/2),2) ,
                '.Yrange "{}", "{}"'.format(np.round(voxel_pos[i,1]-voxel_size/2,2), np.round(voxel_pos[i,1]+voxel_size/2),2),
                '.Zrange "{}", "{}"'.format(np.round(voxel_pos[i,2]-voxel_size/2,2), np.round(voxel_pos[i,2]+voxel_size/2),2),
                '.Create',
            'End With',
            ]
        if i != 0:
            code.append('Solid.Add "component{}:solid{}_0", "component{}:solid{}_{}"'.format(batch+1,batch+1,batch+1,batch+1,i))
        super_code.append('\n'.join(code))
        #print(code)
        # control cst to run the vba code
        #print(i)
    #print('\n'.join(super_code))
    super_code.append('StopVersionStringOverrideMode')    
    super_code.append('Save')
    super_code.append('End Sub')
    
    schematic.execute_vba_code('\n'.join(super_code))

import time
tic = time.time()


# try run again when encountering unexpected errors
def tryrun(voxel_pos,n):
    try:
        runmacro(voxel_pos,n)
    except RuntimeError:
        1/0
        print("Ups nu fejlde det igen")
        tryrun(voxel_pos,n)
    else:
        return 1

#file_path_output = "C:\\Users\\bundg\\OneDrive\\Dokumenter\\7. sem\\Sionna_env\\PEC_PC_Voxel_Grid_0.2.ply"
#file_path_output = "C:\\Users\\bundg\\OneDrive\\Dokumenter\\7. sem\\Sionna_env\\Materials_to_be_used\\PEC_PC_Voxel_Grid_0.5.ply"
header_lines = 19
file_path_output = "C:\\Users\\bundg\\OneDrive\\Dokumenter\\7. sem\\Sionna_env\\Materials_to_be_used\\FR4_PC_Voxel_Grid_0.5.ply"

#min_bound = np.asarray([-100,-100,-50])
min_bound = np.asarray([17,43.5,9.5])
max_bound = np.asarray([20,100,200])
voxel_size = 0.5
data = np.loadtxt(file_path_output, skiprows=header_lines)
# Make output matrix
# 63
number_of_solids_created_at_a_time = 500
number_of_runs = np.ceil((len(data[:,0])-len(data[:,0])%number_of_solids_created_at_a_time)/number_of_solids_created_at_a_time).astype(int)
print(number_of_runs)
for n in tqdm(range(number_of_runs)):
    voxel_pos = min_bound+data[n*number_of_solids_created_at_a_time:(n+1)*number_of_solids_created_at_a_time,0:3]*voxel_size
    tryrun(voxel_pos,n)
new_number_of_solids_created_at_a_time = len(data[:,0])%number_of_solids_created_at_a_time
number_of_solids_created_at_a_time = new_number_of_solids_created_at_a_time
voxel_pos = min_bound+data[-new_number_of_solids_created_at_a_time:,0:3]*voxel_size