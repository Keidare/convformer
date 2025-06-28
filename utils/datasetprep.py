import os
from PIL import Image
import torchvision.transforms as transforms
from itertools import product
from tqdm import tqdm

# Output directory
OUTPUT_DIR = 'data/augmented'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_image_paths(split_dir):
    image_paths = []
    for fname in os.listdir(split_dir):
        if fname.lower().endswith('.tif'):
            image_paths.append(os.path.join(split_dir, fname))
    return image_paths

# Replace hflip with vflip (vertical flip)
def get_augment_combinations():
    return list(product([False, True], repeat=3))  # vflip, rotate, jitter

def augment_and_save(image, save_path, transform_combination):
    vflip, rotate, jitter = transform_combination

    augments = []
    if vflip:
        augments.append(transforms.RandomVerticalFlip(p=1.0))
    if rotate:
        augments.append(transforms.RandomRotation(20))
    if jitter:
        augments.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))

    transform_pipeline = transforms.Compose(augments)
    transformed_image = transform_pipeline(image)
    transformed_image.save(save_path)

# Example usage:
input_dir = './data/magboods_balanced'  # or data_balanced/NO
image_paths = get_image_paths(input_dir)
combinations = get_augment_combinations()

for img_path in tqdm(image_paths, desc="Augmenting"):
    image = Image.open(img_path)
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    for i, combo in enumerate(combinations):
        save_name = f"{base_name}_aug_{i}.tif"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        augment_and_save(image, save_path, combo)
