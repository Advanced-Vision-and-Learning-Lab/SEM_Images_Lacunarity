import matplotlib.pyplot as plt

# Data for the histogram
categories = ['Crystalline Silica (CS)', 'Isocyanate (IPDI)', 'Nickel Oxide (NiO)', 'Silver Nanoparticles (Ag-NP)', 'Untreated']
num_images = [72, 72, 28, 32, 20]

# Create the histogram
plt.figure(figsize=(10, 6))
bars = plt.bar(categories, num_images, color=['lightblue', 'lightgreen', 'red', 'purple', 'orange'])

# Add titles and labels
plt.title('Distribution of Images for Different Categories')
plt.xlabel('Category', fontsize=14, fontweight='bold')
plt.ylabel('Number of Images', fontsize=14, fontweight='bold')

# Rotate category labels if necessary
plt.xticks(rotation=45)

# Add text labels on top of the columns
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, '%d' % int(height), ha='center', va='bottom')

# Show the plot
plt.tight_layout()
plt.show()
