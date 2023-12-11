import gzip
import pickle
import glob

#Find all files with .pkl in data folder
files = glob.glob('data/*/*.pkl')

#Iterate through all files
for file in files:
    #Open file
    with open(file, 'rb') as f_in:
        #Read file
        data = pickle.load(f_in)
        #Create new file name
        new_file = file.replace('.pkl', '.pkl.gz')
        #Open new file
        with gzip.open(new_file, 'wb') as f_out:
            #Write file
            pickle.dump(data, f_out)
