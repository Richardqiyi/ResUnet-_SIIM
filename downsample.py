from PIL import Image
import numpy as np
import os

def downsample_image(image_path, output_path, new_size=(512, 512), is_mask=False):
    with Image.open(image_path) as img:
        img_resized = img.resize(new_size, Image.ANTIALIAS)

        if is_mask:
            img_array = np.array(img_resized)
            img_array[img_array > 128] = 255
            img_array[img_array <= 128] = 0
            img_resized = Image.fromarray(img_array)

        img_resized.save(output_path)

def process_folder(source_folder, target_folder, is_mask=False):
    os.makedirs(target_folder, exist_ok=True)

    for file_name in os.listdir(source_folder):
        file_path = os.path.join(source_folder, file_name)
        output_path = os.path.join(target_folder, file_name)

        downsample_image(file_path, output_path, is_mask=is_mask)

base_dir = '/code/ResUnet/SIIM_test'
input_folder = os.path.join(base_dir, 'input')
output_folder = os.path.join(base_dir, 'output')
input_1_folder = os.path.join(base_dir, 'input_1')
output_1_folder = os.path.join(base_dir, 'output_1')

process_folder(input_folder, input_1_folder)
process_folder(output_folder, output_1_folder, is_mask=True)

