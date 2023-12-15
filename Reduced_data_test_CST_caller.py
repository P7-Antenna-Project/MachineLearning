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
        im.append('StoreParameter("{}", {})'.format(par_name[i], par_value[0][i]))
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

    schematic.execute_vba_code(code)


def try_run(S11_path, paraname, para):
    try:
        run_parameters_in_cst(S11_path, paraname, para)
    except RuntimeError:
        try_run(S11_path, paraname, para)
    else:
        return 1



if __name__ == "__main__":

    laptop_path = r"C:\Program Files (x86)\CST Studio Suite 2023\AMD64\python_cst_libraries"
    sys.path.append(laptop_path)

    import cst.interface

    # Import handles
    cstfile = r"C:\Users\nlyho\OneDrive - Aalborg Universitet\7. semester\Git\MachineLearning\CST files\Wire_antenna_simple_2.cst"
    DE = cst.interface.DesignEnvironment()
    microwavestructure = DE.open_project(cstfile)
    modeler = microwavestructure.modeler
    schematic = microwavestructure.schematic
    
    WIRE = True

    if WIRE:
        paraname = ['wire_length','wire_height','wire_thickness']
    else:
        paraname = []

    with open("Reduced_data_Test/Wire_reduced_data_inverse2_pred.pkl", "rb") as f:
        data = pickle.load(f)

    for idx, pars in tqdm(enumerate(data["Predictions"])):
        tic = time.time()
        if WIRE:
            S11_pat = f"C:/Users/nlyho/OneDrive - Aalborg Universitet/7. semester/Git/MachineLearning/Reduced_data_Test/50_size_cst_curves_WIRE/S11_{idx}.txt"
        else:
            S11_pat = f"C:/Users/nlyho/OneDrive - Aalborg Universitet/7. semester/Git/MachineLearning/Reduced_data_Test/50_size_cst_curves_MIFA/S11_{idx}.txt"
        print(pars)
        try_run(S11_pat, paraname, pars)
        toc = time.time()
        print(f'{toc-tic} used for this run')

   


