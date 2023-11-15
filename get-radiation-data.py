import numpy as np
import pandas as pd
import sys
from tqdm import tqdm

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

# path til cst-filen:
#cst_path =  r"C:\Users\madsl\Dropbox\AAU\EIT 7. sem\P7\Python6_stuff\MachineLearning\CST files\Wire_antpyenna_simple_2.cst" (NOT USED)
def represent(para):
    return para/(ns-1)*(para_max-para_min)+para_min

para = np.zeros((num, dim_x))
para[:, -1] = np.arange(num)
for j in range(dim_x):
    para[:, j] = (np.mod(para[:, -1], np.prod(ns[j:]))/np.prod(ns[j+1:])).astype(int)
para = represent(para)
pd.DataFrame(para).to_csv(f'data\\distanceTest{num}.csv', header=None, index=None)


new_cstfile = r"C:\Users\nlyho\OneDrive - Aalborg Universitet\7. semester\Git\MachineLearning\CST files\Wire_antenna_simple_2.cst"
DE = cst.interface.DesignEnvironment()
microwavestructure = DE.open_project(new_cstfile)
modeler = microwavestructure.modeler
schematic = microwavestructure.schematic
start = 0
num = 5
# generate vba code that changes the parameters in cst
def setpara(paraname, paravalue, num=dim_x):
    code = ''
    for i in range(0,num):
        code += 'StoreParameter("{}", {})'.format(paraname[i], paravalue[i])
        if i < num - 1:
            code += '\n'
    return code

# generate the complete vba code
def runmacro(phi0,phi45,phi90, theta0, theta45, theta90, s11file, s21file, paraname, paravalue, lenpara): #commented s21file 
    code = [
        'Option Explicit',
        'Sub Main',
        # empty the results so that you do not mix the old and new results
        'DeleteResults',
        # change the parameters
        setpara(paraname, paravalue, lenpara),
        'Rebuild',
        # execute the solver
        'Solver.Start',
        # select the result to export

        'SelectTreeItem("Tables\\1D Results\\Directivity,Phi=0.0")', # S Change this to the correct port HUSK HUSKL HUSK HUSK HUSLK HUSK
         # export the selected result
         'With ASCIIExport',
         '.Reset',
         '.Filename("{}")'.format(phi0),
         '.Execute',
         'End With',

        'SelectTreeItem("Tables\\1D Results\\Directivity,Phi=45.0")', # S Change this to the correct port HUSK HUSKL HUSK HUSK HUSLK HUSK
         # export the selected result
         'With ASCIIExport',
         '.Reset',
         '.Filename("{}")'.format(phi45),
         '.Execute',
         'End With',
        
        'SelectTreeItem("Tables\\1D Results\\Directivity,Phi=90.0")', # S Change this to the correct port HUSK HUSKL HUSK HUSK HUSLK HUSK
         # export the selected result
         'With ASCIIExport',
         '.Reset',
         '.Filename("{}")'.format(phi90),
         '.Execute',
         'End With',

        'SelectTreeItem("Tables\\1D Results\\Directivity,Theta=0.0")', # S Change this to the correct port HUSK HUSKL HUSK HUSK HUSLK HUSK
         # export the selected result
         'With ASCIIExport',
         '.Reset',
         '.Filename("{}")'.format(theta0),
         '.Execute',
         'End With',

        'SelectTreeItem("Tables\\1D Results\\Directivity,Theta=45.0")', # S Change this to the correct port HUSK HUSKL HUSK HUSK HUSLK HUSK
         # export the selected result
         'With ASCIIExport',
         '.Reset',
         '.Filename("{}")'.format(theta45),
         '.Execute',
         'End With',

        'SelectTreeItem("Tables\\1D Results\\Directivity,Theta=90.0")', # S Change this to the correct port HUSK HUSKL HUSK HUSK HUSLK HUSK
         # export the selected result
         'With ASCIIExport',
         '.Reset',
         '.Filename("{}")'.format(theta90),
         '.Execute',
         'End With',



        ################################# this is for s11 params
         # S Change this to the correct port HUSK HUSKL HUSK HUSK HUSLK HUSK
         'SelectTreeItem("1D Results\\S-Parameters\\S1,1")', # S Change this to the correct port HUSK HUSKL HUSK HUSK HUSLK HUSK
         # export the selected result
         'With ASCIIExport',
         '.Reset',
         '.Filename("{}")'.format(s11file),
         '.Execute',
         'End With',

        ################################ this is for ff cuts
        #'sItem = Resulttree.GetTreeResults("Farfields","farfield","",treepaths,resulttypes,filenames,resultinformation)', 
        #'SelectTreeItem("Farfields\\Farfield Cuts\\Excitation [1]\\Phi=0")', 
        #'With ASCIIExport',
        #'.Reset', 
        #'.FileName ("{}"+"radiation"+Str(num)+".txt")'.format(folder_radiation), 
        #'.Execute', 
        #'End With',

        ################################## comment this when doing s21 ############################################
        #  # select the result to export
        #  'SelectTreeItem("1D Results\\S-Parameters\\S2,1")',
        #  # export the selected result
        #   'With ASCIIExport',
        #   '.Reset',
        #   '.Filename("{}")'.format(s21file),
        #   '.Execute',
        #   'End With',
        'Save',
        'End Sub'
    ]
    code = '\n'.join(code)
    # print(code)
    # control cst to run the vba code
    schematic.execute_vba_code(code)

import time
tic = time.time()


# try run again when encountering unexpected errors
def tryrun(paraname, para, dim, s11file, s21file,phi0,phi45,phi90, theta0, theta45, theta90):
    try:
        runmacro(phi0,phi45,phi90, theta0, theta45, theta90,s11file, s21file, paraname=paraname, paravalue=para, lenpara=dim)
    except RuntimeError:
        tryrun(paraname, para, dim, s11file, s21file,phi0,phi45,phi90, theta0, theta45, theta90)
    else:
        return 1

angles = [0,45,90]

# sweep and running simulation and collect data
for i in range(start, num):
    print(f'Run {i+1}/{num}:')
    toc = time.time()
    s11file =f'C:/Users/nlyho/OneDrive - Aalborg Universitet/7. semester/Git/MachineLearning/data/wireAntennaSimple2Results_nicolai/test_s11/s11_{i}.txt'
    
    # for j in angles:
    #     theta = f'data//radiation//theta//theta{j}_{i}.txt'
    #     phi = f'data//radiation//phi//phi{j}_{i}.txt'

# "C:\Users\madsl\Dropbox\AAU\EIT 7. sem\P7\Python6_stuff\data\wireAntennaSimple2Results\theta"

    phi0 = f'C:/Users/nlyho/OneDrive - Aalborg Universitet/7. semester/Git/MachineLearning/data/wireAntennaSimple2Results_nicolai/test_phi/phi0_{i}.txt'
    phi45= f'C:/Users/nlyho/OneDrive - Aalborg Universitet/7. semester/Git/MachineLearning/data/wireAntennaSimple2Results_nicolai/test_phi/phi45_{i}.txt'
    phi90 = f'C:/Users/nlyho/OneDrive - Aalborg Universitet/7. semester/Git/MachineLearning/data/wireAntennaSimple2Results_nicolai/test_phi/phi90_{i}.txt'
    theta0 = f'C:/Users/nlyho/OneDrive - Aalborg Universitet/7. semester/Git/MachineLearning/data/wireAntennaSimple2Results_nicolai/test_theta/theta0_{i}.txt'
    theta45 = f'C:/Users/nlyho/OneDrive - Aalborg Universitet/7. semester/Git/MachineLearning/data/wireAntennaSimple2Results_nicolai/test_theta/theta45_{i}.txt'
    theta90 = f'C:/Users/nlyho/OneDrive - Aalborg Universitet/7. semester/Git/MachineLearning/data/wireAntennaSimple2Results_nicolai/test_theta/theta90_{i}.txt'
    
    #r'C:/Users/nlyho/OneDrive - Aalborg Universitet/7. semester/CSTEnv/data/s11/s11file.txt'
    # f'C:/Users/nlyho/OneDrive - Aalborg Universitet/7. semester/CSTEnv/data/s11/s11file_{i}.txt'
    
    s21file =f'data//s21//s21_{i}.txt'
    tryrun(paraname, para[i], dim_x, s11file, s21file,phi0,phi45,phi90, theta0, theta45, theta90)
    print(para[i])
    print(f'{time.time()-toc} used for this run, {time.time()-tic} used, {(time.time()-tic)/(i+1-start)*(num-i-1)} more needed')