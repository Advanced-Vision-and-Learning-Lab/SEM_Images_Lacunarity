import matplotlib.pyplot as plt
import numpy as np

# Data for the histogram
categories = ['Crystalline Silica (CS)', 'Isocyanate (IPDI)', 'Nickel Oxide (NiO)', 'Silver Nanoparticles (Ag-NP)', 'Untreated']
train_num_images = [80, 80, 16, 16, 32]  # Number of images in the train dataset
test_num_images = [32, 32, 16, 16, 16]    # Number of images in the test dataset

# Define pastel colors
pastel_colors = ['#A6CEE3', '#B2DF8A', '#FDBF6F', '#FFCCFF', '#FF9896']

# Width of each bar
bar_width = 0.35

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars for train dataset
train_bars = ax.bar(np.arange(len(categories)), train_num_images, width=bar_width, color=pastel_colors, label='Train Dataset')

# Plot bars for test dataset
test_bars = ax.bar(np.arange(len(categories)) + bar_width, test_num_images, width=bar_width, color=pastel_colors, alpha=0.5, label='Test Dataset')

# Add titles and labels
ax.set_title('Distribution of Images for Different Categories')
ax.set_xlabel('Category', fontsize=16, fontweight='bold')
ax.set_ylabel('Number of Images', fontsize=14, fontweight='bold')

# Set x-axis ticks and labels
ax.set_xticks(np.arange(len(categories)) + bar_width / 2)
ax.set_xticklabels(categories, rotation=45, fontsize=12)

# Add text labels on top of the columns for train dataset
for bar in train_bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, height, '%d' % int(height), ha='center', va='bottom')

# Add text labels on top of the columns for test dataset
for bar in test_bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, height, '%d' % int(height), ha='center', va='bottom')

# Add legend
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()
