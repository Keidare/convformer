import os
import shutil

val_dir = './data/tiny-imagenet-200/val/images'
val_annotations = './data/tiny-imagenet-200/val/val_annotations.txt'
target_dir = './data/tiny-imagenet-200/val/organized/'

# Create target directory structure
os.makedirs(target_dir, exist_ok=True)

# Read annotations
with open(val_annotations, 'r') as f:
    data = f.readlines()

for line in data:
    fields = line.split('\t')
    img_file, class_id = fields[0], fields[1]
    
    class_dir = os.path.join(target_dir, class_id)
    os.makedirs(class_dir, exist_ok=True)
    
    src = os.path.join(val_dir, img_file)
    dst = os.path.join(class_dir, img_file)
    shutil.move(src, dst)
