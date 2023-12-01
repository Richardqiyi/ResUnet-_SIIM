import os
import shutil
import random
base_dir = '/code/ResUnet/SIIM-ACR-Pneumothorax-Seg-XR'
images_dir = os.path.join(base_dir, 'images')
masks_dir = os.path.join(base_dir, 'masks')

output_dirs = {
    'train_input': os.path.join(base_dir, 'train/input'),
    'valid_input': os.path.join(base_dir, 'valid/input'),
    'train_output': os.path.join(base_dir, 'train/output'),
    'valid_output': os.path.join(base_dir, 'valid/output')
}
for path in output_dirs.values():
    os.makedirs(path, exist_ok=True)

images = os.listdir(images_dir)
masks = os.listdir(masks_dir)

assert sorted(images) == sorted(masks)

split_index = int(len(images) * 0.8)

train_images = images[:split_index]
valid_images = images[split_index:]

for image in train_images:
    shutil.copy(os.path.join(images_dir, image), output_dirs['train_input'])
    shutil.copy(os.path.join(masks_dir, image), output_dirs['train_output'])

for image in valid_images:
    shutil.copy(os.path.join(images_dir, image), output_dirs['valid_input'])
    shutil.copy(os.path.join(masks_dir, image), output_dirs['valid_output'])

