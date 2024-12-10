import os
import json
import numpy as np
import cv2
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split


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
        if f.endswith(".json") and not f.startswith(".")
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

def get_num_labels(labels_path):
    return len(
        [
            f
            for f in os.listdir(labels_path)
            if f.endswith(".json") and not f.startswith(".")
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


# Function to split dataset into train, validation, and test sets
def split_augmented_dataset(
    augmented_images_dir,
    augmented_masks_dir,
    output_dir,
    train_ratio=0.8,
    val_ratio=0.1,
):
    # Get all images and masks
    images = sorted(Path(augmented_images_dir).glob("*.jpg"))
    masks = sorted(Path(augmented_masks_dir).glob("*.png"))

    # Ensure images and masks match
    assert len(images) == len(masks), "Number of images and masks do not match!"
    for img, mask in zip(images, masks):
        assert img.stem == mask.stem, f"Mismatch: {img.stem} and {mask.stem}"

    # Split dataset into train and temp (val + test)
    train_images, temp_images, train_masks, temp_masks = train_test_split(
        images, masks, test_size=(1 - train_ratio), random_state=42
    )

    # Calculate the ratio for validation and test from the remaining data
    val_ratio_adjusted = val_ratio / (1 - train_ratio)  # Adjust for remaining data (10%)
    
    # Split temp into val and test
    val_images, test_images, val_masks, test_masks = train_test_split(
        temp_images, temp_masks, test_size=(1 - val_ratio_adjusted), random_state=42
    )

    # Define output directories
    splits = {
        "train": (train_images, train_masks),
        "val": (val_images, val_masks),
        "test": (test_images, test_masks),
    }

    for split, (split_images, split_masks) in splits.items():
        split_image_dir = Path(output_dir) / split / "images"
        split_mask_dir = Path(output_dir) / split / "masks"
        split_image_dir.mkdir(parents=True, exist_ok=True)
        split_mask_dir.mkdir(parents=True, exist_ok=True)

        # Copy files to respective directories
        for img, mask in zip(split_images, split_masks):
            shutil.copy(img, split_image_dir / img.name)
            shutil.copy(mask, split_mask_dir / mask.name)

    print(f"Dataset split into train, val, and test sets in {output_dir}")
    print("apapapapappa")

def correct_image_rotation(input_dir, output_dir):
    """
    Corrects the rotation of car images to ensure all cars are horizontal.

    Parameters:
        input_dir (str): Directory containing input images.
        output_dir (str): Directory to save corrected images.
    """
    os.makedirs(output_dir, exist_ok=True)
    print('Hello')

    for image_path in Path(input_dir).glob("*.jpg"):
        # Load the image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not read the image at {image_path}. Skipping.")
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Create a binary mask: Threshold to separate the car from the background
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Get the minimum area rectangle that bounds the largest contour
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            # Extract the angle from the rectangle
            angle = rect[-1]

            # Correct the angle to ensure horizontal alignment
            if angle < -45:
                angle = 90 + angle
            elif angle > 45:
                angle = angle - 90

            # Rotate the image
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

            # Save the corrected image
            output_path = os.path.join(output_dir, image_path.name)
            cv2.imwrite(output_path, rotated)
            print(f"Saved corrected image: {output_path}")
        else:
            print(f"No contours found in {image_path}. Skipping.")


