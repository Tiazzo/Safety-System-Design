import os
import json
import numpy as np
from pathlib import Path


# Check if every images in the dataset has a corresponding label
def check_images_labels(images_path, labels_path):
    images = os.listdir(images_path)
    labels = os.listdir(labels_path)
    images = {
        os.path.splitext(f)[0]
        for f in os.listdir(images_path)
        if f.endswith(".jpg") and not f.startswith(".")
    }
    labels = {
        os.path.splitext(f)[0]
        for f in os.listdir(labels_path)
        if f.endswith(".txt") and not f.startswith(".")
    }
    if set(images) == set(labels):
        print("All images have corresponding labels")
    else:
        print("Some images don't have corresponding labels")
        print("Images without labels: ", set(images) - set(labels))
        print("Labels without images: ", set(labels) - set(images))


# Retrieve number of images in the dataset
def get_num_images(images_path):
    return len(
        [
            f
            for f in os.listdir(images_path)
            if f.endswith(".jpg") and not f.startswith(".")
        ]
    )


# Convert from LabelMe format to COCO format
def labelme_to_coco(labelme_dir, output_file):
    """
    Converts LabelMe JSON files to COCO format.

    Parameters:
        labelme_dir (str): Directory containing LabelMe JSON files.
        output_file (str): Path to save the COCO formatted JSON file.
    """
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "window"}],  # Define your categories here
    }

    annotation_id = 1  # Unique ID for each annotation
    image_id = 1  # Unique ID for each image

    for json_file in Path(labelme_dir).glob("*.json"):
        with open(json_file, "r") as f:
            labelme_data = json.load(f)

        # Add image information
        coco_format["images"].append(
            {
                "id": image_id,
                "file_name": labelme_data["imagePath"],
                "width": labelme_data["imageWidth"],
                "height": labelme_data["imageHeight"],
            }
        )

        # Add annotations for this image
        for shape in labelme_data["shapes"]:
            points = np.array(shape["points"])
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            width = x_max - x_min
            height = y_max - y_min

            # Flatten segmentation points
            segmentation = points.flatten().tolist()

            # Add annotation
            coco_format["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,  # Category ID (1 for "window")
                    "segmentation": [segmentation],
                    "area": width * height,
                    "bbox": [x_min, y_min, width, height],
                    "iscrowd": 0,
                }
            )

            annotation_id += 1

        image_id += 1

    # Save COCO formatted JSON
    with open(output_file, "w") as f:
        json.dump(coco_format, f, indent=4)

    print(f"COCO annotations saved to {output_file}")
