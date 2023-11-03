import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
import csv

paraname = ['wire_length','wire_height', 'wire_thickness']

dim_x = len(paraname)
# dim_y = 51
para_min = np.array([157/10,4,0.5])
para_max = np.array([157,15,3])
para_step = np.array([7.43684210526316, 1, 0.25])
ns = np.array([(para_max[0]-para_min[0])/para_step[0]+1,(para_max[1]-para_min[1])/para_step[1]+1,(para_max[2]-para_min[2])/para_step[2]+1]).astype(np.int) #Number of parameter samples
num = np.prod(ns)

laptop_path = r'C:\Program Files (x86)\CST Studio Suite 2021\AMD64\python_cst_libraries'
sys.path.append(laptop_path)
import cst.interface

save_path = r'data'
cst_path =  r'C:\Users\nlyho\Desktop\Simple_wire_2'
def represent(para):
    return para/(ns-1)*(para_max-para_min)+para_min

para = np.zeros((num, dim_x))
para[:, -1] = np.arange(num)
for j in range(dim_x):
    para[:, j] = (np.mod(para[:, -1], np.prod(ns[j:]))/np.prod(ns[j+1:])).astype(np.int)
para = represent(para)
pd.DataFrame(para).to_csv(f'C:/Users/nlyho/Desktop/Simple_wire_2/data{num}.csv', header=None, index=None)

# Define handles to CST
new_cstfile = cst_path+'\\Wire_antenna_simple_2.cst'
DE = cst.interface.DesignEnvironment()
microwavestructure = DE.open_project(new_cstfile)
modeler = microwavestructure.modeler
schematic = microwavestructure.schematic

# generate vba code that changes the parameters in cst
def setpara(paraname, paravalue, num=dim_x):
    code = ''
    for i in range(num):
        code += 'StoreParameter("{}", {})'.format(paraname[i], paravalue[i])
        if i < num - 1:
            code += '\n'
    return code


# generate the complete vba code
def runmacro(s11file, s21file, paraname, paravalue, lenpara):
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
        'SelectTreeItem("1D Results\\S-Parameters\\S1,1")', # Change this to the correct port
        # export the selected result
        'With ASCIIExport',
        '.Reset',
        '.Filename("{}")'.format(s11file),
        '.Execute',
        'End With',
        # # select the result to export
        # 'SelectTreeItem("1D Results\\S-Parameters\\S2,1")',
        # # export the selected result
        # 'With ASCIIExport',
        # '.Reset',
        # '.Filename("{}")'.format(s21file),
        # '.Execute',
        # 'End With',
        'Save',
        'End Sub'
    ]
    code = '\n'.join(code)
    # print(code)
    # control cst to run the vba code
    schematic.execute_vba_code(code)


import time
tic = time.time()
start = 0

# try run again when encountering unexpected errors
def tryrun(paraname, para, dim, s11file, s21file):
    try:
        runmacro(s11file, s21file, paraname=paraname, paravalue=para, lenpara=dim)
    except RuntimeError:
        tryrun(paraname, para, dim, s11file, s21file)
    else:
        return 1


# sweep and running simulation and collect data
run_time = np.zeros(num)
for i in range(start, num):
    print(f'Run {i+1}/{num}:')
    toc = time.time()
    s11file =f"C:/Users/nlyho/Desktop/Simple_wire_2/Results/s11file_{i}.txt"
    
    s21file = f'C:/Users/nlyho/OneDrive - Aalborg Universitet/7. semester/CSTEnv/data/s21/s21file_{i}.txt'
    tryrun(paraname, para[i], dim_x, s11file, s21file)

    #Use this if you want to save the run times in a csv file
    run_time[i] = time.time()-toc
    pd.DataFrame(run_time).to_csv(f'C:/Users/nlyho/Desktop/Simple_wire_2/run_times.csv', header=None, index=None)
                        
    print(f'{time.time()-toc} used for this run, {time.time()-tic} used, {(time.time()-tic)/(i+1-start)*(num-i-1)} more needed')
    
    