import os
import zipfile
import shutil
from ultralytics import YOLO
import requests

# --- Configuration ---

# 1. Dataset Source
ROBOFLOW_ZIP_URL = None 
# Use 'r' before the string to handle backslashes correctly in Windows
LOCAL_ZIP_FILENAME = r"C:\Users\yemij\Downloads\FRC_pre_test_2026.v3i.yolov8.zip"

# 2. Image Settings
# This MUST match the 'Resize: Stretch to' value from Roboflow (640 for v3)
IMAGE_SIZE_FOR_TRAINING = 640 

# 3. Training Parameters
EPOCHS = 50 
BATCH_SIZE = 8  # Keep small for CPU/Laptop. Increase to 16 if you have a GPU.
TRAINING_RUN_NAME = 'frc_fuel_auto_train'

# 4. Directories (These were missing in your error)
UNZIP_DIR = 'roboflow_dataset_extracted'
MODELS_DIR = 'models'

# --- Functions ---

def download_file(url, filename):
    print(f"Attempting to download {url} to {filename}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename} successfully.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False

def find_data_yaml(directory):
    for root, dirs, files in os.walk(directory):
        if 'data.yaml' in files:
            return os.path.join(root, 'data.yaml')
    return None

def main():
    # --- Step 1: Locate and Unzip the dataset ---
    zip_path = LOCAL_ZIP_FILENAME
    
    if not os.path.exists(zip_path):
        print(f"Error: LOCAL ZIP file not found at: {zip_path}")
        print("Please verify the path in the script matches your download location.")
        return

    print(f"\nUnzipping {zip_path} to {UNZIP_DIR}...")
    os.makedirs(UNZIP_DIR, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(UNZIP_DIR)
        print("Dataset unzipped successfully.")
    except Exception as e:
        print(f"Error unzipping file: {e}")
        return

    # --- Step 2: Find data.yaml ---
    data_yaml_path = find_data_yaml(UNZIP_DIR)
    if not data_yaml_path:
        print(f"Error: data.yaml not found in {UNZIP_DIR}. Check dataset structure.")
        return
    print(f"Found data.yaml at: {data_yaml_path}")

    # --- Step 3: Train the YOLOv8 model ---
    print(f"\nStarting YOLOv8 model local training on {data_yaml_path}...")
    print(f"Training for {EPOCHS} epochs with image size {IMAGE_SIZE_FOR_TRAINING} and batch size {BATCH_SIZE}...")
    try:
        # Load a pre-trained YOLOv8n model
        model = YOLO('yolov8n.pt') 

        results = model.train(data=data_yaml_path, 
                              epochs=EPOCHS, 
                              imgsz=IMAGE_SIZE_FOR_TRAINING, 
                              batch=BATCH_SIZE,  
                              name=TRAINING_RUN_NAME) 

        print("\nTraining complete!")
        
        # --- Step 4: Copy best.pt to the designated models folder ---
        output_dir = results.save_dir 
        best_model_path = os.path.join(output_dir, 'weights', 'best.pt')
        
        os.makedirs(MODELS_DIR, exist_ok=True)
        final_model_path = os.path.join(MODELS_DIR, 'best.pt')

        shutil.copy(best_model_path, final_model_path)
        print(f"Copied trained model (best.pt) to: {final_model_path}")
        print("\nYour YOLOv8 model is ready for inference!")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        print("Check if you have enough memory or if the data.yaml path is correct.")

if __name__ == "__main__":
    main()