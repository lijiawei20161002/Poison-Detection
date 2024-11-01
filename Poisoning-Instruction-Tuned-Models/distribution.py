import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

def read_txt_file(file_path):
    idx = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r"(\d+):\s*(-?\d+\.\d+)", line)
            if match:
                idx.append(int(match.group(1)))
    return idx

# File paths
file_path = 'influence_scores.csv'
idx_path = 'identified_poisons.txt'

# Read data
data = pd.read_csv(file_path)
ids = read_txt_file(idx_path)

# Separate the full dataset and the highlighted subset
highlighted_data = data.iloc[ids]

# Plot the full distribution
plt.rcParams['font.family'] = 'Comic Sans MS'  
plt.rcParams['font.size'] = 18
plt.figure(figsize=(10, 6)) 
bins = np.linspace(-50, 50, 50)  
plt.hist(data['influence_score'], bins=bins, color='mediumseagreen', edgecolor='black', alpha=0.7, label='All Data')
plt.hist(highlighted_data['influence_score'], bins=bins, color='darkviolet', label='Poisons')

# Add labels and title
plt.title('Distribution of Influence Scores (After Transformation)')
plt.xlabel('Influence Score')
plt.ylabel('Frequency')
plt.legend()

# Display grid and save plot
plt.grid()
plt.savefig("distribution_highlighted.png")