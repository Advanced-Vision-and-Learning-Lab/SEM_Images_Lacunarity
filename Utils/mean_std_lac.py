import os
import csv
import numpy as np
import re
import pdb
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu
from scipy.stats import shapiro

# Path to the CSV file where the results are saved
csv_file_path = 'C:/Users/aksha/Peeples_Lab/SEM_Images_Lacunarity/Datasets/DC_changed_lacunarity_results.csv'

# Path to the CSV file where the class-wise statistics will be saved
class_stats_csv_file_path = 'C:/Users/aksha/Peeples_Lab/SEM_Images_Lacunarity/Datasets/DC_changed_lacunarity_class_stats.csv'

# Define classes
classes = {
    "Crystalline Silica (CS)": [],
    "Isocyanate (IPDI)": [],
    "Nickel Oxide (NiO)": [],
    "Silver Nanoparticles (Ag-NP)": [],
    "Untreated": []
}

# Function to extract numerical value from tensor string
def extract_numerical_value(tensor_str):
    match = re.search(r"\[([\d.e-]+)\]", tensor_str)
    if match:
        return float(match.group(1))
    return None

# Read the results from the CSV file and organize them by class
with open(csv_file_path, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        image_path = row["Image Path"]
        base_lacunarity_str = row["Base_Lacunarity"]
        dbc_lacunarity_str = row["DBC_Lacunarity"]
        
        # Extract numerical values from tensor strings
        base_lacunarity = float(base_lacunarity_str)
        dbc_lacunarity = float(dbc_lacunarity_str)
        
        # Determine the class based on the image path
        if "Crystalline Silica (CS)" in image_path:
            class_name = "Crystalline Silica (CS)"
        elif "Isocyanate (IPDI)" in image_path:
            class_name = "Isocyanate (IPDI)"
        elif "Nickel Oxide (NiO)" in image_path:
            class_name = "Nickel Oxide (NiO)"
        elif "Silver Nanoparticles (Ag-NP)" in image_path:
            class_name = "Silver Nanoparticles (Ag-NP)"
        else:
            class_name = "Untreated"
        
        # Append the lacunarity values to the corresponding class
        classes[class_name].append((base_lacunarity, dbc_lacunarity))

# Calculate the mean and standard deviation for each class
class_stats = {}
for class_name, lacunarity_values in classes.items():
    base_lacunarity_values, dbc_lacunarity_values = zip(*lacunarity_values)
    base_shapiro = shapiro(base_lacunarity_values)
    dbc_shapiro = shapiro(dbc_lacunarity_values)
    base_lacunarity_mean = np.mean(base_lacunarity_values)
    dbc_lacunarity_mean = np.mean(dbc_lacunarity_values)
    base_lacunarity_std = np.std(base_lacunarity_values)
    dbc_lacunarity_std = np.std(dbc_lacunarity_values)
    
    class_stats[class_name] = {
        "base_shapiro": base_shapiro,
        "dbc_shapiro": dbc_shapiro,
        "Base_Lacunarity_Mean": base_lacunarity_mean,
        "Base_Lacunarity_Std": base_lacunarity_std,
        "DBC_Lacunarity_Mean": dbc_lacunarity_mean,
        "DBC_Lacunarity_Std": dbc_lacunarity_std
    }

# Write the class-wise statistics to a CSV file
with open(class_stats_csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Class", "base_shapiro", "dbc_shapiro", "Base_Lacunarity_Mean", "Base_Lacunarity_Std", "DBC_Lacunarity_Mean", "DBC_Lacunarity_Std"])
    for class_name, stats in class_stats.items():
        writer.writerow([class_name, stats["base_shapiro"], stats["dbc_shapiro"], stats["Base_Lacunarity_Mean"], stats["Base_Lacunarity_Std"], stats["DBC_Lacunarity_Mean"], stats["DBC_Lacunarity_Std"]])

print("Class-wise statistics saved to CSV file.")
