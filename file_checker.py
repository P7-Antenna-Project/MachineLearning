import os

# Define the directory where the files are located
directory = r"data\wireAntennaSimple2Results_inc_eff/test_eff"
filename = "tot_eff_"


# Get a list of all files in the directory
all_files = os.listdir(directory)

# Create a list of all the expected file names
expected_files = [f"{filename}{i}.txt" for i in range(2508)]

# Check if each expected file is in the list of actual files
missing_files = [file for file in expected_files if file not in all_files]

if missing_files:
    print("The following files are missing:", missing_files)
else:
    print("All files are present.")