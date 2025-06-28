import os
import pandas as pd

image_dir = 'data/augmented/test'  # Folder with TIFF images
output_csv = 'data/ground_test.csv'

data = []

for filename in os.listdir(image_dir):
    if filename.endswith('.tif'):
        parts = filename.split('-')
        if len(parts) >= 2:
            second_part = parts[1].strip().upper()
            if "NO" in second_part:
                label = 0
            elif "YES" in second_part:
                label = 1
            else:
                print(f"⚠️  Skipping: No NO/YES in second part → {filename}")
                continue
            data.append({'filename': filename, 'label': label})
        else:
            print(f"⚠️  Skipping: Unexpected filename format → {filename}")

df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)

# Count labels
label_counts = df['label'].value_counts().sort_index()

print(f"📊 Label counts:")
print(f"  0 (NO):  {label_counts.get(0, 0)}")
print(f"  1 (YES): {label_counts.get(1, 0)}")
