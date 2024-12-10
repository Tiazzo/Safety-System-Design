import os
import shutil
import pandas as pd
import cv2
from collections import defaultdict
import random
import json
from pathlib import Path
from sklearn.model_selection import train_test_split


def select_images_by_category(dataset_dir, output_dir, total_images=400):
    """
    Selects a total of `total_images` distributed equally across categories
    (first letter in filenames) and saves them to the specified output directory.

    Parameters:
        dataset_dir (str): Directory containing the images.
        output_dir (str): Directory to save selected images.
        total_images (int): Total number of images to select.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Group images by category
    categories = defaultdict(list)
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".jpg"):
            category = filename.split("_")[
                0
            ]  # Category based on the first part of the name
            categories[category].append(filename)

    # Calculate the number of images per category
    total_categories = len(categories)
    images_per_category = total_images // total_categories

    # Select images
    selected_images = []
    for category, files in categories.items():
        random.shuffle(files)
        selected_images.extend(files[:images_per_category])

    # Ensure exactly `total_images`
    if len(selected_images) < total_images:
        extra_needed = total_images - len(selected_images)
        remaining_files = [
            f
            for category, files in categories.items()
            for f in files
            if f not in selected_images
        ]
        random.shuffle(remaining_files)
        selected_images.extend(remaining_files[:extra_needed])

    # Copy selected images to the output directory
    for filename in selected_images:
        shutil.copy(
            os.path.join(dataset_dir, filename), os.path.join(output_dir, filename)
        )

    print(f"Selected {len(selected_images)} images. Saved in '{output_dir}'.")


def split_dataset(
    labeled_dir, train_dir, val_dir, test_dir, test_size=0.3, val_split=0.5
):
    """
    Splits a labeled dataset into training, validation, and test sets.

    Parameters:
        labeled_dir (str): Directory containing labeled images.
        train_dir (str): Directory to save training images.
        val_dir (str): Directory to save validation images.
        test_dir (str): Directory to save test images.
        test_size (float): Proportion of the dataset to include in the test and validation split.
        val_split (float): Proportion of the test/validation split to assign to validation.
    """
    # Get all image files
    images = sorted([f for f in os.listdir(labeled_dir) if f.endswith(".jpg")])

    # Split dataset
    train_images, val_test_images = train_test_split(
        images, test_size=test_size, random_state=42
    )
    val_images, test_images = train_test_split(
        val_test_images, test_size=val_split, random_state=42
    )

    # Move files to respective directories
    def move_files(file_list, src_dir, dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
        for file_name in file_list:
            shutil.copy(os.path.join(src_dir, file_name), dest_dir)

    move_files(train_images, labeled_dir, train_dir)
    move_files(val_images, labeled_dir, val_dir)
    move_files(test_images, labeled_dir, test_dir)

    print(
        f"Dataset split complete: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}."
    )


def separate_files(source_dir, images_dir, json_dir):
    """
    Separates .jpg and .json files from the source directory into respective directories.

    Parameters:
        source_dir (str): Directory containing the files.
        images_dir (str): Directory to save .jpg files.
        json_dir (str): Directory to save .json files.
    """
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    for filename in os.listdir(source_dir):
        if filename.endswith(".jpg"):
            shutil.move(
                os.path.join(source_dir, filename), os.path.join(images_dir, filename)
            )
        elif filename.endswith(".json"):
            shutil.move(
                os.path.join(source_dir, filename), os.path.join(json_dir, filename)
            )

    print(f"All files have been moved: Images -> '{images_dir}', JSON -> '{json_dir}'.")


# Function to cleanup the path of JSON files
def clean_image_path(labelme_dir):
    """
    Cleans and splits the `imagePath` field in LabelMe JSON files, removing unnecessary directory paths.

    Parameters:
        labelme_dir (str): Directory containing LabelMe JSON files.
    """
    # Iterate over all JSON files in the directory
    for json_file in Path(labelme_dir).glob("*.json"):
        try:
            # Load the JSON file
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Get the original imagePath
            original_path = data.get("imagePath", "")

            # Use split on '\\' to extract the file name if needed
            if "\\" in original_path:
                new_path = original_path.split("\\")[-1]
                data["imagePath"] = new_path  # Update the imagePath field

                # Save the updated JSON back to the file
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)

                print(
                    f"Updated imagePath in {json_file.name}: '{original_path}' -> '{new_path}'"
                )
            else:
                print(
                    f"No changes needed for {json_file.name} (imagePath: '{original_path}')"
                )
        except Exception as e:
            print(f"Failed to process {json_file.name}: {e}")


def select_additional_images(
    dataset_dir, output_dir, previous_images_dir, total_images=400
):
    """
    Selects a total of `total_images` distributed equally across categories
    (first letter in filenames) while excluding previously selected images.
    Saves the new images to the specified output directory.

    Parameters:
        dataset_dir (str): Directory containing the original dataset.
        output_dir (str): Directory to save the newly selected images.
        previous_images_dir (str): Directory containing previously selected images.
        total_images (int): Total number of new images to select.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect previously selected images
    previously_selected = set(os.listdir(previous_images_dir))

    # Group images by category while excluding previously selected ones
    categories = defaultdict(list)
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".jpg") and filename not in previously_selected:
            category = filename.split("_")[
                0
            ]  # Category based on the first part of the name
            categories[category].append(filename)

    # Calculate the number of images per category
    total_categories = len(categories)
    images_per_category = total_images // total_categories

    # Select new images
    selected_images = []
    for category, files in categories.items():
        random.shuffle(files)
        selected_images.extend(files[:images_per_category])

    # Ensure exactly `total_images`
    if len(selected_images) < total_images:
        extra_needed = total_images - len(selected_images)
        remaining_files = [
            f
            for category, files in categories.items()
            for f in files
            if f not in selected_images
        ]
        random.shuffle(remaining_files)
        selected_images.extend(remaining_files[:extra_needed])

    # Copy selected images to the output directory
    for filename in selected_images:
        shutil.copy(
            os.path.join(dataset_dir, filename), os.path.join(output_dir, filename)
        )

    print(f"Selected {len(selected_images)} new images. Saved in '{output_dir}'.")


def manual_labeling(images_dir, output_csv):
    """
    Manually label car orientation as 'left' or 'right'.

    Parameters:
        images_dir (str): Directory containing images to label.
        output_csv (str): Path to save the labels as a CSV file.
    """
    images = sorted(Path(images_dir).glob("*.jpg"))
    labels = []

    for image_path in images:
        # Display the image
        image = cv2.imread(str(image_path))
        cv2.imshow("Label this image (Press 'l' for left, 'r' for right)", image)

        # Wait for user input
        key = cv2.waitKey(0)  # Wait indefinitely for key press
        if key == ord("l"):
            labels.append({"image": image_path.name, "label": "left"})
        elif key == ord("r"):
            labels.append({"image": image_path.name, "label": "right"})
        elif key == ord("q"):
            print("Exiting manual labeling...")
            break  # Exit labeling if 'q' is pressed

        # Close the image window
        cv2.destroyAllWindows()

    # Save labels to CSV
    pd.DataFrame(labels).to_csv(output_csv, index=False)
    print(f"Labels saved to {output_csv}")


def main():
    """
    Main function to execute operations based on user choice.
    """
    print("Choose an operation:")
    print("1. Select images by category")
    print("2. Split dataset into train, validation, and test")
    print("3. Separate images (.jpg) and annotations (.json)")
    print("4. Clean image paths in JSON files")
    print("5. Manually label orientation images")
    print("6. Select additional images")
    choice = input("Enter your choice (1, 2, 3, 4, 5, 6): ")

    if choice == "1":
        dataset_dir = "data/processed/images"
        output_dir = "data/train/images"
        total_images = 400
        select_images_by_category(dataset_dir, output_dir, total_images)
    elif choice == "2":
        labeled_dir = "data/labeled/images"
        train_dir = "data/train"
        val_dir = "data/val"
        test_dir = "data/test"
        split_dataset(labeled_dir, train_dir, val_dir, test_dir)
    elif choice == "3":
        source_dir = (
            "/Users/mattiacarlino/Politecnico coding/Safety-System-Design/images"
        )
        images_dir = "/Users/mattiacarlino/Politecnico coding/Safety-System-Design/data/with_labels/images"
        json_dir = "/Users/mattiacarlino/Politecnico coding/Safety-System-Design/data/with_labels/json"
        separate_files(source_dir, images_dir, json_dir)
    elif choice == "4":
        labelme_dir = "data/with_labels/json"
        clean_image_path(labelme_dir)
    elif choice == "5":
        images_dir = "data/split/train/images"
        output_csv = "data/split/labels.csv"
        manual_labeling(images_dir, output_csv)
    elif choice == "6":
        dataset_dir = "data/processed/images"
        output_dir = "data/train/new_images"
        previous_images_dir = "data/train/images"
        total_images = 400
        select_additional_images(
            dataset_dir, output_dir, previous_images_dir, total_images
        )
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()
