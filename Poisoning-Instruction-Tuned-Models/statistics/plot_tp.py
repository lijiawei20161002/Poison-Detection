import matplotlib.pyplot as plt

# Read data from the file
data_file = "task_poisons.txt"
tasks = []
tp_ratios = []

with open(data_file, "r") as file:
    for line in file:
        parts = line.strip().split(", TP: ")
        task_name = parts[0].split(":")[0].strip()
        tp_ratio = float(parts[1])
        tasks.append(task_name)
        tp_ratios.append(tp_ratio)

# Plot the data
plt.figure(figsize=(10, 6))
plt.barh(tasks, tp_ratios, color="purple")

# Customize the plot style
plt.title("True Positive Ratios for Critical Poison Detection Across Tasks", fontsize=14, fontname="Comic Sans MS")
plt.xlabel("True Positive Ratio (TP)", fontsize=12, fontname="Comic Sans MS")
plt.xticks(fontsize=10, fontname="Comic Sans MS")
plt.yticks(fontsize=10, fontname="Comic Sans MS")
plt.tight_layout()

# Adjust x-axis limits to ensure labels fit
max_ratio = max(tp_ratios)
plt.xlim(0, max_ratio + 0.05)  # Add padding to the right

# Annotate the bar plot with TP ratios
for i, v in enumerate(tp_ratios):
    plt.text(v + 0.005, i, f"{v*100:.1f}%", va='center', fontsize=10, fontname="Comic Sans MS")

# Save and show the plot
plt.grid()
plt.savefig("tp_ratios_plot.png")