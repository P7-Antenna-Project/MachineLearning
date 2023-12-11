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
result_path = "/data/Step_files"

# HUSK AT SÆTTE START OG STOP ( LINJE 57+59 )

paraname = ['wire_length','wire_height', 'wire_thickness']


dim_x = len(paraname)
# dim_y = 51
para_min = np.array([157/10,4,0.5])
para_max = np.array([157,15,3])
para_step = np.array([7.43684210526316, 1, 0.25])
ns = np.array([(para_max[0]-para_min[0])/para_step[0]+1,(para_max[1]-para_min[1])/para_step[1]+1,(para_max[2]-para_min[2])/para_step[2]+1]).astype(int) #Number of parameter samples
num = np.prod(ns)


laptop_path = r"C:\Program Files (x86)\CST Studio Suite 2023\AMD64\python_cst_libraries"
sys.path.append(laptop_path)
import cst.interface
import shutil


def represent(para):
    return para/(ns-1)*(para_max-para_min)+para_min

para = np.zeros((num, dim_x))
para[:, -1] = np.arange(num)
for j in range(dim_x):
    para[:, j] = (np.mod(para[:, -1], np.prod(ns[j:]))/np.prod(ns[j+1:])).astype(int)
para = represent(para)
pd.DataFrame(para).to_csv(f'data\\par_comb{num}.csv', header=None, index=None)

#new_cstfile = r"C:\Users\nlyho\OneDrive - Aalborg Universitet\7. semester\Git\MachineLearning\CST files\Wire_antenna_simple_2.cst"
new_cstfile = f"{path}\CST files\Wire_antenna_simple_2.cst"


DE = cst.interface.DesignEnvironment()
microwavestructure = DE.open_project(new_cstfile)
modeler = microwavestructure.modeler
schematic = microwavestructure.schematic



# --------------------------------- DEFFINING THE RUNS ---------------------------------
# Set this if you want to start at a specific run
start = 2008 #int(num-1)
# Set this num if you want to stop at a specific run
num = int(num)
# -------------------------------------------------------------------------------------



# generate vba code that changes the parameters in cst
def setpara(paraname, paravalue, num=dim_x):
    code = ''
    for i in range(0,num):
        code += 'StoreParameter("{}", {})'.format(paraname[i], paravalue[i])
        if i < num - 1:
            code += '\n'
    return code

# generate the complete vba code
def runmacro(paraname, paravalue, lenpara, antenna_file, antenna_solid_path): #commented s21file 
    code = [
        'Option Explicit',
        'Sub Main',
        # empty the results so that you do not mix the old and new results
        'DeleteResults',
        # change the parameters
        setpara(paraname, paravalue, lenpara),
        'Rebuild',
        'SelectTreeItem("{}")'.format(antenna_solid_path), # S Change this to the correct port HUSK HUSKL HUSK HUSK HUSLK HUSK
        'With STEP',
        '.Reset',
        '.FileName("{}")'.format(antenna_file),
        '.WriteSelectedSolids',
        'End With',
        'Save',
        'End Sub'
    ]
    code = '\n'.join(code)
    #print(code)
    # control cst to run the vba code
    schematic.execute_vba_code(code)

import time
tic = time.time()


# try run again when encountering unexpected errors
def tryrun(paraname, para, dim, A_File, A_solid_path):
    try:
        runmacro(paraname=paraname, paravalue=para, lenpara=dim, antenna_file=A_File, antenna_solid_path = A_solid_path)
    except RuntimeError:
        1/0
        print("Ups nu fejlde det igen")
        tryrun(paraname, para, dim, A_File, A_solid_path)
    else:
        return 1


# sweep and running simulation and collect data
# Don't change this for loop, change the start and num variables at the top
for i in range(start, num):
    print(f'Run {i+1}/{num}:')
    toc = time.time()

# -------------------------------------- CHANGE TO DIR --------------------------------------

# "C:\Users\madsl\Dropbox\AAU\EIT 7. sem\P7\Python6_stuff\data\wireAntennaSimple2Results\theta"
    Antenna_file = f'{path}{result_path}/Antenna_{i}_step.step'
    antenna_solid_path = "Components\\component1\\wire_antenna"

# -------------------------------------------------------------------------------------------    
    
    #r'C:/Users/nlyho/OneDrive - Aalborg Universitet/7. semester/CSTEnv/data/s11/s11file.txt'
    # f'C:/Users/nlyho/OneDrive - Aalborg Universitet/7. semester/CSTEnv/data/s11/s11file_{i}.txt'
    
    tryrun(paraname, para[i], dim_x, Antenna_file, antenna_solid_path)
    print(para[i])
    print(f'{time.time()-toc} used for this run, {time.time()-tic} used, {(time.time()-tic)/(i+1-start)*(num-i-1)} more needed')