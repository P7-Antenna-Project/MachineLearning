import numpy as np
import pandas as pd
import sys
import pickle
import time
from tqdm import tqdm
import shutil


def run_parameters_in_cst(s11file, par_name, par_value):
    im = []
    for i in range(len(par_name)):
        im.append('StoreParameter("{}", {})'.format(par_name[i], par_value[i]))
    code = [
        "Option Explicit",
        'Sub Main',
        'DeleteResults',
        *im,
        'Rebuild',
        'Solver.Start',
        'SelectTreeItem("1D Results\\S-Parameters\\S1,1")', # S Change this to the correct port HUSK HUSKL HUSK HUSK HUSLK HUSK
         # export the selected result
         'With ASCIIExport',
         '.Reset',
         '.Filename("{}")'.format(s11file),
         '.Execute',
         'End With',
         'Save',
         'End Sub'
    ]
    code = '\n'.join(code)
    start_time = time.perf_counter()
    schematic.execute_vba_code(code)
    run_time.append(time.perf_counter() - start_time)

def try_run(S11_path, paraname, para):
    try:
        run_parameters_in_cst(S11_path, paraname, para)
    except RuntimeError:
        try_run(S11_path, paraname, para)
    else:
        return 1



if __name__ == "__main__":
    WIRE_ANTENNA = False
    run_time = []

    laptop_path = r"C:\Program Files (x86)\CST Studio Suite 2023\AMD64\python_cst_libraries"
    sys.path.append(laptop_path)

    import cst.interface

    # Import handles
    if WIRE_ANTENNA:
        cstfile = r"C:\Users\nlyho\OneDrive - Aalborg Universitet\7. semester\Git\MachineLearning\CST files\Wire_antenna_simple_2.cst"
    else:
        cstfile = r"C:\Users\nlyho\OneDrive - Aalborg Universitet\7. semester\Git\MachineLearning\CST files\MIFAinmilimeters.cst"
    DE = cst.interface.DesignEnvironment()
    microwavestructure = DE.open_project(cstfile)
    modeler = microwavestructure.modeler
    schematic = microwavestructure.schematic
    

    if WIRE_ANTENNA:
        paraname = ['wire_length','wire_height','wire_thickness']
        with open("Reduced_data_Test/Wire_testing_inverse2_pred.pkl", "rb") as f:
            data = pickle.load(f)
    else:
        paraname = ['Tw1','groundingPinTopLength','Line1_height','substrateH']
        with open("Reduced_data_Test/MIFA_testing_inverse2_pred.pkl", "rb") as f:
            data = pickle.load(f)

   

    for idx, pars in tqdm(enumerate(data["Predictions"])):
        tic = time.time()
        if WIRE_ANTENNA:
            S11_path = f"C:/Users/nlyho/OneDrive - Aalborg Universitet/7. semester/Git/MachineLearning/Reduced_data_Test/S11_wire/S11_{idx}.txt"
        else:
            S11_path = f"C:/Users/nlyho/OneDrive - Aalborg Universitet/7. semester/Git/MachineLearning/Reduced_data_Test/S11_MIFA/S11_{idx}.txt"
        print(pars)
        try_run(S11_path, paraname, pars)
        toc = time.time()
        print(f'{toc-tic} used for this run')
        
    if WIRE_ANTENNA:
        np.savetxt("Reduced_data_Test/Wire_run_time.txt", run_time)
    else:
        np.savetxt("Reduced_data_Test/MIFA_run_time.txt", run_time)


