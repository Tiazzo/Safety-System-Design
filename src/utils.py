import os

# Check if every images in the dataset has a corresponding label
def check_images_labels(images_path, labels_path):

# Filter out hidden files and non-image files
    images = os.listdir(images_path)
    labels = os.listdir(labels_path)
    images = {os.path.splitext(f)[0] for f in os.listdir(images_path) if f.endswith('.jpg') and not f.startswith('.')}
    labels = {os.path.splitext(f)[0] for f in os.listdir(labels_path) if f.endswith('.txt') and not f.startswith('.')}
    if set(images) == set(labels):
        print("All images have corresponding labels")
    else:
        print("Some images don't have corresponding labels")
        print("Images without labels: ", set(images) - set(labels))
        print("Labels without images: ", set(labels) - set(images))
    