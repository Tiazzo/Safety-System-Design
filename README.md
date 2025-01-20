# Safety-System-Design 🚗
### Group E -  Mattia Carlino, Mikael Motin, Lorenzo Paravano, Yusheng Yang

### Introduction
This project was conducted in collaboration with Autoliv to automate the labeling process of side-view car windows using image segmentation

### Before you get started
- It is **recommended** to use a Python virtual environment (venv) for libraries and packages
  - To create a venv with necessary libraries and packages, run the following file:
  - `/notebooks/installation_setup.py` 

- If you want to train the models (U-Net & YOLOv11) please download the **images** and **labels** from [the  GP22 dataset](https://zenodo.org/records/6366808)  containing 1,480 pictures of cars with corresponding feature labels
 
### Available notebooks
*(If you want to train the models, you need to **first** run the `preprocessing notebook`)*

#### Preprocessing & Data Augmentation
The following notebook contains the necessary steps to prepare the dataset for training. Steps included: `Removing background`, `Flipping cars`, `Scaling cars`, `Data augmentation`
- `preprocessing`

#### Model training and prediction
- `yolo_train_and_predict` - train the *YOLO model* and/or predict with the trained model
- `unet_train_and_predict` - train the *U-Net model* and/or predict with the trained model

#### Convert to CAD Output
- `output_in_cad`- will convert predicted masks to CAD format (.DXF)

#### Model evaluation
The following notebooks are designed to evaluate the performance of both models using a set of 100 images and will find the best and worst prediction in the set.
- `evaluate_yolo`
- `evaluate_unet`

- Metrics used:
- `Mean Hausdorff Distance` `Mean IoU` `Mean Dice` `Mean Precision` `Mean Recall`  `Mean MAE` `Mean F1-Score`