import json
import os
import shutil

from tqdm import tqdm

def convertCOCOtoYOLO(jsonPath, outputPath, categories):
    print(f"Reading JSON from: {jsonPath}")
    print(f"Output directory: {outputPath}")

    # If the output directory exists, delete it
    if os.path.exists(outputPath):
        shutil.rmtree(outputPath)

    # Create the new output directory
    os.makedirs(outputPath)

    with open(jsonPath) as f:
        data = json.load(f)

    images = data["images"]

    # Set up the progress bar
    with tqdm(total=len(images), desc="Converting COCO to YOLO", unit="image") as pbar:
        for image in images:
            imageID = image["id"]
            fileName = image["file_name"]

            # Intercept the path and remove any leading directory
            fileName = os.path.basename(fileName)  # Keeps only the file name, removing directories

            width = image["width"]
            height = image["height"]

            annotations = [
                ann
                for ann in data["annotations"]
                if ann["image_id"] == imageID and 0 <= ann["category_id"] <= categories
            ]

            outputFilePath = os.path.join(outputPath, f"{os.path.splitext(fileName)[0]}.txt")

            # Ensure the directory for the output file exists
            outputFileDir = os.path.dirname(outputFilePath)
            if not os.path.exists(outputFileDir):
                os.makedirs(outputFileDir)

            if annotations:
                with open(outputFilePath, "w") as f:
                    for ann in annotations:
                        categoryID = ann["category_id"]
                        bbox = ann["bbox"]
                        xCenter = (bbox[0] + bbox[2] / 2) / width
                        yCenter = (bbox[1] + bbox[3] / 2) / height
                        w = bbox[2] / width
                        h = bbox[3] / height
                        f.write(f"{categoryID} {xCenter} {yCenter} {w} {h}\n")

            # Update the progress bar
            pbar.update(1)

convertCOCOtoYOLO(r"val_annotations.json", r"data1/val/labels", 1)
