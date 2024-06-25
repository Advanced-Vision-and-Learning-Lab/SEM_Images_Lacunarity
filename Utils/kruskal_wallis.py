import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

# Load the CSV file
csv_file_path = 'C:/Users/aksha/Peeples_Lab/SEM_Images_Lacunarity/Datasets/DC_changed_lacunarity_results_all.csv'
df = pd.read_csv(csv_file_path)



# Extract class information from the image path and map to full class names
df['Class'] = df['Class']


# Shortened class names for plotting
short_class_names = {
    "Crystalline Silica (CS)": "CS",
    "Isocyanate (IPDI)": "IPDI",
    "Nickel Oxide (NiO)": "NiO",
    "Silver Nanoparticles (Ag-NP)": "Ag-NP",
    "Untreated": "Untreated"
}


# Perform Kruskal-Wallis H-test for Base Lacunarity
base_lacunarity_groups = [group for _, group in df.groupby('Class')['Base_Lacunarity']]
h_statistic_base, p_value_base = stats.kruskal(*base_lacunarity_groups)

# Perform Kruskal-Wallis H-test for DBC Lacunarity
dbc_lacunarity_groups = [group for _, group in df.groupby('Class')['DBC_Lacunarity']]
h_statistic_dbc, p_value_dbc = stats.kruskal(*dbc_lacunarity_groups)

# Perform Kruskal-Wallis H-test for DBC Lacunarity
multi_lacunarity_groups = [group for _, group in df.groupby('Class')['Multi_Lacunarity']]
h_statistic_multi, p_value_multi = stats.kruskal(*multi_lacunarity_groups)

print("Kruskal-Wallis H-test results:")
print(f"Base Lacunarity - H-statistic: {h_statistic_base:.4f}, p-value: {p_value_base:.4f}")
print(f"DBC Lacunarity - H-statistic: {h_statistic_dbc:.4f}, p-value: {p_value_dbc:.4f}")
print(f"Multi Lacunarity - H-statistic: {h_statistic_multi:.4f}, p-value: {p_value_multi:.4f}")


# Visualize the distributions
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
sns.boxplot(x='Class', y='Base_Lacunarity', data=df)
plt.title('Base Lacunarity Distribution by Class')
plt.xticks(ticks=range(len(short_class_names)), labels=list(short_class_names.values()), rotation=45)

plt.subplot(1, 3, 2)
sns.boxplot(x='Class', y='DBC_Lacunarity', data=df)
plt.title('DBC Lacunarity Distribution by Class')
plt.xticks(ticks=range(len(short_class_names)), labels=list(short_class_names.values()), rotation=45)

plt.subplot(1, 3, 3)
sns.boxplot(x='Class', y='Multi_Lacunarity', data=df)
plt.title('Multi Lacunarity Distribution by Class')
plt.xticks(ticks=range(len(short_class_names)), labels=list(short_class_names.values()), rotation=45)

plt.tight_layout()
plt.show()

# If the Kruskal-Wallis test is significant, perform post-hoc pairwise comparisons
if p_value_base < 0.05:
    print("\nPost-hoc pairwise comparisons for Base Lacunarity:")
    classes = df['Class'].unique()
    for i in range(len(classes)):
        for j in range(i+1, len(classes)):
            class1 = classes[i]
            class2 = classes[j]
            stat, p = stats.mannwhitneyu(
                df[df['Class'] == class1]['Base_Lacunarity'],
                df[df['Class'] == class2]['Base_Lacunarity']
            )
            print(f"{class1} vs {class2}: p-value = {p:.4f}")

if p_value_dbc < 0.05:
    print("\nPost-hoc pairwise comparisons for DBC Lacunarity:")
    classes = df['Class'].unique()
    for i in range(len(classes)):
        for j in range(i+1, len(classes)):
            class1 = classes[i]
            class2 = classes[j]
            stat, p = stats.mannwhitneyu(
                df[df['Class'] == class1]['DBC_Lacunarity'],
                df[df['Class'] == class2]['DBC_Lacunarity']
            )
            print(f"{class1} vs {class2}: p-value = {p:.4f}")