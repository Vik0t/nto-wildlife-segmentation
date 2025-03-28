import cv2
import numpy as np
import os
from PIL import Image
from gray_finder import is_grey_scale
from tqdm import tqdm
import glob
import json
# Function
# to adjust brightness and contrast
def adjust_brightness_contrast(image, alpha=1.0, beta=0):
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

# Function to apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_image

# Function to apply Gaussian Blur for noise reduction
def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    return blurred

# Function to resize the image while maintaining aspect ratio
def resize_image(image, target_width=640):
    height, width = image.shape[:2]
    aspect_ratio = width / height
    new_height = int(target_width / aspect_ratio)
    
    resized = cv2.resize(image, (target_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized

# Function to apply Unsharp Masking for edge sharpening
def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

# 
def preprocess_image(image_path, output_path, *args):
    """
    Function to process a single image
    """
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    enhanced_image = image
    #for arg in args:
    #    enhanced_image = arg(enhanced_image)
    ##enhanced_image = apply_clahe(image, clip_limit=8.0, tile_grid_size=(8, 8)) 
    ##enhanced_image = cv2.convertScaleAbs(enhanced_image, alpha=3.5, beta=0) 
    #enhanced_image= cv2.fastNlMeansDenoising(enhanced_image, 10,3,7,31)
    #enhanced_image = apply_clahe(enhanced_image, clip_limit=5, tile_grid_size=(8, 8))
    ##enhanced_image = apply_gaussian_blur(enhanced_image, kernel_size=(5, 5), sigma=10)
    #enhanced_image = unsharp_mask(enhanced_image, amount=5)
    cv2.imwrite(output_path, enhanced_image)
    #print(f"Processed image saved to {output_path}")


def preprocess_images_in_directory(input_dir, gray_dir, color_dir):
    """
    Function to process all images in a directory
    """
    if not os.path.exists(gray_dir):
        os.makedirs(gray_dir)
    if not os.path.exists(color_dir):
        os.makedirs(color_dir)

    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)    
            output_path = os.path.join(gray_dir, filename) \
            if is_grey_scale(input_dir+"/"+filename) else  os.path.join(color_dir, filename)
            preprocess_image(input_path, output_path)

def preprocess_images_in_directory_json(input_dir, gray_dir, color_dir, path_json):
    """
    Function to process all images in a directory
    """
    gray_data={"images":[],"annotations":[]}
    color_data={"images":[],"annotations":[]}
    with open(path_json) as f:
        data=json.load(f)
    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            for i in range(len(data['images'])):
                if data['images'][i]['file_name']==filename:
                    fid=data['images'][i]['id']
                    pos1=i
                    break
            for j in range(len(data['annotations'])):
                if data['annotations'][j]['id']==fid:
                    pos2=j
                    break

            input_path = os.path.join(input_dir, filename)  

            if is_grey_scale(input_dir+"/"+filename):
                output_path = os.path.join(gray_dir, filename)
                gray_data['images'].append(data['images'][pos1])
                gray_data['annotations'].append(data['annotations'][pos2])
            else:
                color_data['images'].append(data['images'][pos1])
                color_data['annotations'].append(data['annotations'][pos2])
                output_path=os.path.join(color_dir, filename)

            preprocess_image(input_path, output_path) #коммент если не надо раскидывать картинки

    with json.open("gray.json", "w") as file: 
        file.write(json.dumps(gray_data, indent=2))
    with json.open("color.json", "w") as file: 
        file.write(json.dumps(color_data, indent=2))

def assign_images_to_json(input_dir, path_json, json_name):
    img_data={"images":[],"annotations":[]}

    with open(path_json) as f:
        data=json.load(f)
    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            for i in range(len(data['images'])):
                if data['images'][i]['file_name']==filename:
                    fid=data['images'][i]['id']
                    pos1=i
                    break
            for j in range(len(data['annotations'])):
                if data['annotations'][j]['image_id']==fid:
                    pos2=j
                    break

            img_data['images'].append(data['images'][pos1])
            img_data['annotations'].append(data['annotations'][pos2])

            
    with open(json_name, "w") as file: 
        file.write(json.dumps(img_data, indent=2))
            


# Define input and output directories
input_directory = '/home/vik0t/hackaton/segment-anything-2/data1/val/images' 
color_directory = '/home/vik0t/hackaton/train_split/color' 
gray_directory = '/home/vik0t/hackaton/train_split/gray'
json_path = '/home/vik0t/hackaton/val_annotations.json'
# Preprocess all images in the input directory
assign_images_to_json(input_directory, json_path, "/home/vik0t/hackaton/segment-anything-2/gray_val.json")