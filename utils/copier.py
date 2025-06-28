import os
import csv

# Path to the augmented image directory
augmented_dir = 'data/augmented'
output_csv = 'data/GT_Final.csv'

# Create list of (filename, label) pairs
data = []

# Counters
count_yes = 0  # label 1
count_no = 0   # label 0

for fname in os.listdir(augmented_dir):
    if fname.lower().endswith('.tif'):
        parts = fname.split('-')
        if len(parts) > 1 and 'NO' in parts[1].upper():
            label = 0
            count_no += 1
        else:
            label = 1
            count_yes += 1
        data.append((fname, label))

# Write CSV
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filename', 'label'])  # Header
    writer.writerows(data)

print(f"Ground truth saved to {output_csv} with {len(data)} entries.")
print(f"Total YES (1): {count_yes}")
print(f"Total NO  (0): {count_no}")
