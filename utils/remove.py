import os,csv

# Define the directory where your files are located
directory = './data/bones/'

# Define the keyword you're looking for in the second element of the delimited filenames
keyword = 'YES'

# Create a list to hold the results
results = [["filename", "label"]]  # Add header row

# Counters for labels
count_yes = 0
count_no = 0

# Iterate over each file in the directory
for filename in os.listdir(directory):
    # Split the filename by the delimiter '-'
    parts = filename.split('-')
    # Check if the second element matches the keyword
    if len(parts) > 1 and keyword in parts[1]:
        # If it does, append the filename and 1 to the results list
        results.append([filename, 1])
        count_yes += 1
    else:
        # If it doesn't, append the filename and 0 to the results list
        results.append([filename, 0])
        count_no += 1

# Write the results to a CSV file
with open('ground.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(results)

# Print the counts of 1s and 0s
print("Count of YES (1):", count_yes)
print("Count of NO (0):", count_no)


# import os
# from PIL import Image
# from torchvision import transforms
# from pathlib import Path
# from itertools import combinations

# # Define transformations with explicit names
# transformations = {
#     "horizontal_flip": transforms.RandomHorizontalFlip(p=1.0),
#     "vertical_flip": transforms.RandomVerticalFlip(p=1.0),
#     "random_rotate": transforms.RandomRotation(degrees=20)  # ±20 degrees rotation
# }

# # Directory to save transformed images
# save_dir = "transformed_images"
# os.makedirs(save_dir, exist_ok=True)

# def apply_transformations(image, transform_list):
#     """Apply a list of transformations sequentially to an image."""
#     for transform in transform_list:
#         image = transform(image)
#     return image

# def save_transformed_images_from_folder(input_folder, save_dir):
#     """Apply transformations to all .tif images in a folder and save results."""
#     image_files = list(Path(input_folder).glob("*.tif"))

#     for image_path in image_files:
#         try:
#             image = Image.open(image_path).convert("RGB")  # Ensure compatibility

#             # Apply individual transformations
#             for name, transform in transformations.items():
#                 transformed_image = transform(image)
#                 transformed_image.save(os.path.join(save_dir, f"{image_path.stem}_{name}.tif"))

#             # Apply combinations of transformations (ORDER FIXED TO AVOID DUPLICATES)
#             for r in range(2, len(transformations) + 1):
#                 for combo in combinations(sorted(transformations.items()), r):  # Sorting fixes order
#                     combo_names, combo_transforms = zip(*combo)
#                     transformed_image = apply_transformations(image.copy(), combo_transforms)
#                     combo_name = "_".join(combo_names)
#                     transformed_image.save(os.path.join(save_dir, f"{image_path.stem}_{combo_name}.tif"))

#         except Exception as e:
#             print(f"Error processing {image_path}: {e}")

# # Example usage
# input_folder = "./data/bones"
# save_transformed_images_from_folder(input_folder, save_dir)

# # Remove all files with '_' in the filename in the bones folder
# for filename in os.listdir(directory):
#     if '_' in filename:
#         file_path = os.path.join(directory, filename)
#         if os.path.isfile(file_path):
#             os.remove(file_path)

# Get the list of YES and NO files based on the delimited filename
yes_files = []
no_files = []

for filename in os.listdir(directory):
    parts = filename.split('-')  # Split the filename by '-'
    if len(parts) > 1:  # Ensure there are enough parts after splitting
        if "YES" in parts[1]:  # Check the second part for "YES"
            yes_files.append(filename)
        elif "NO" in parts[1]:  # Check the second part for "NO"
            no_files.append(filename)

# # Remove NO files until their count matches the count of YES files
# while len(no_files) > len(yes_files):
#     file_to_remove = no_files.pop()  # Remove a file from the NO list
#     file_path = os.path.join(directory, file_to_remove)
#     if os.path.isfile(file_path):
#         os.remove(file_path)

# Print the results
print("YES files:", len(yes_files))
print("NO files:", len(no_files))