import re
import numpy as np
import os
import pandas as pd


folder_path = './Kappa_VB/' # or 
#folder_path = '~/nisargq/entropy_production/postprocessing'
#folder_path = '~/entropy_production/mpi_dynamik/Kappa_VB'
file_type = 'kappa_Lyap_qpdrvn_NNN_J2byJ1_'

J21_ratios = {0.002: [], 0.1: [],  0.25: [],  0.4: [], 0.5: [],  0.625: [],  0.8: [],  0.99: [] }

print(os.getcwd())
print(os.path.exists(folder_path))
print(os.path.isdir(folder_path))
# check if path exists
if os.path.exists(folder_path):
    all_files = os.listdir(folder_path)

all_files.sort()

prefix1, suffix = 'kappa_Lyap_qpdrvn_NNN_J2byJ1_', '.txt'
prefix2 = 'kappa_Lyap_qpdrvn_NNN_J1byJ2_'

# store based on the J21 ratio and create a Series and maybe a pandas DF
for file_name in all_files:
    #print(file_name)
    if file_name.startswith(prefix1): 
    # same thing but with regex: 
    # if re.match(rf'^kappa_Lyap_qpdrvn_NNN_J2byJ1_', file_name):  # Match files starting with the prefix
        file_name = file_name.removeprefix(prefix1)
        if file_name.endswith(suffix):
            file_name = file_name.removesuffix(suffix)
        #new_name = file_name.removesuffix(".txt")
        print(file_name)

        if file_name.startswith("0pt002"):
            J21_ratios[0.002].append(file_name) #filtered_files.append(new_name)
        if file_name.startswith("0pt100"):
            J21_ratios[0.1].append(file_name) #filtered_files.append(new_name)
        if file_name.startswith("0pt250"):
            J21_ratios[0.25].append(file_name) #filtered_files.append(new_name)
        if file_name.startswith("0pt400"):
            J21_ratios[0.4].append(file_name) #filtered_files.append(new_name)
        if file_name.startswith("0pt500"):
            J21_ratios[0.5].append(file_name) #filtered_files.append(new_name)
        if file_name.startswith("0pt625"):
            J21_ratios[0.625].append(file_name) #filtered_files.append(new_name)
        if file_name.startswith("0pt800"):
            J21_ratios[0.8].append(file_name) #filtered_files.append(new_name)
        if file_name.startswith("0pt990"):
            J21_ratios[0.99].append(file_name) #filtered_files.append(new_name)
        
        # alternative with regex
        # re.sub(r'^kappa_qpdrvn', '', file_name)
        # filtered_files.append(new_name)

#print(J21_ratios[0.002])

df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in J21_ratios.items()]))

# Print the DataFrame
print("Dictionary as a DataFrame:")
print(df)
print(prefix1, '\n', suffix)
#files = ["0pt100_1024configs_200", "0pt100_104configs_200", "0pt100_51configs_200"]
for k,v in J21_ratios.items():
    # Sort based on the integer in the "configs" part
    sorted_files = sorted(v, key=lambda filnam: int(filnam.split("_")[1][:-7]))
    #print(sorted_files)
    restored_filenames = [prefix1 + x + suffix for x in sorted_files]
    if len(restored_filenames):
        print(restored_filenames[-1]) 
        # remember: folder_path = './Kappa_VB'
        with  open(os.path.join(folder_path,restored_filenames[-1])) as f:
            print(f.read())


J_norm = 1 + np.sqrt(np.array([0.990, 0.625, 0.4, 0.25, 0.1, 0.002])**2)
# J1 = Jcos(theta); J2 = Jsin(theta)
V =  np.array([2.720, 2.783, 2.756, 2.798, 2.668, 2.644])
V_norm = V/J_norm 
print(V_norm)
J_add = 1 + np.sqrt(np.array([0.1, 0.25, 0.4, 0.5, 0.625, 0.8, 0.99])**2)
V_add = np.array([1.351, 1.455, 1.604, 1.778, 2.028, 2.5507, 2.730])
V_norm0 = V_add/J_add 
V_fin = np.concatenate((V_norm0, V_norm))

print("V_normalized for tan(theta): \n ", V_fin)
