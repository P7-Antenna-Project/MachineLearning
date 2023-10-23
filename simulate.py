import numpy as np
import pandas as pd

# name of parameters to sweep
paraname = ['Offset2mid', 'TransformPAR']

# number of parameters to sweep
dim = 2

# number of settings of parameters to sweep: 729=9*9*9
num = 112
n = 

# min and max values of parameters to sweep
para_min = np.array([10, 2, 8])
para_max = np.array([14, 6, 12])


import sys

# replace with your cst path here
laptop_path = r'C:\Program Files (x86)\CST Studio Suite 2021\AMD64\python_cst_libraries'
sys.path.append(laptop_path)
import cst.interface

path = r'C:\Users\nlyho\OneDrive - Aalborg Universitet\7. semester\Git\Project\CST_RTX'

new_cstfile = path+'\\Anders_Version_To_Sweep.cst'
DE = cst.interface.DesignEnvironment()
microwavestructure = DE.open_project(new_cstfile)
modeler = microwavestructure.modeler
schematic = microwavestructure.schematic


# generate vba code that changes the parameters in cst
def setpara(paraname, paravalue, num=dim):
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
        'FDSolver.Start',
        # select the result to export
        'SelectTreeItem("1D Results\\S-Parameters\\S1,1")',
        # export the selected result
        'With ASCIIExport',
        '.Reset',
        '.Filename("{}")'.format(s11file),
        '.Execute',
        'End With',
        # select the result to export
        'SelectTreeItem("1D Results\\S-Parameters\\S2,1")',
        # export the selected result
        'With ASCIIExport',
        '.Reset',
        '.Filename("{}")'.format(s21file),
        '.Execute',
        'End With',
        'Save',
        'End Sub'
    ]
    code = '\n'.join(code)
    # control cst to run the vba code
    schematic.execute_vba_code(code)


# denormalize the parameters
def represent(para):
    return para/(n-1)*(para_max-para_min)+para_min


# generate the list of parameters to sweep
pararecord = np.zeros((num, dim))
para = np.zeros(dim)
for i in range(num):
    for j in range(dim):
        para[j] = int(np.mod(i, n**(dim-j))/n**(dim-j-1))
    pararecord[i] = represent(para)


# save the list of parameters to a csv file
pd.DataFrame(pararecord).to_csv(path+f'data\\pararecord{num}.csv', header=None, index=None)

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
for i in range(start, num):
    print(f'Run {i+1}/{num}:')
    toc = time.time()
    s11file = path+f'data\\s11files{num}\\s11file_{i}.txt'
    s21file = path+f'data\\s21files{num}\\s21file_{i}.txt'
    tryrun(paraname, pararecord[i], dim, s11file, s21file)
    print(f'{time.time()-toc} used for this run, {time.time()-tic} used, {(time.time()-tic)/(i+1-start)*(num-i-1)} more needed')