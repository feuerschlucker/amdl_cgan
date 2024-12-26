import os
import cv2
import numpy as np
import scipy.io as sio
from keras.utils import to_categorical

# Paths
image_dir = '/home/hroethl/amdl_cgan/data/train_cropped/images'
label_file = '/home/hroethl/amdl_cgan/data/train_cropped/train.mat'  


output_size = (28, 50)  # Target size for images
cutoff = 800  # Minimum pixel area

# Load labels from .mat file
mat_data = sio.loadmat(label_file)
print(mat_data)
labels = mat_data['y'].flatten()  # Adjust key based on .mat structure
print(labels.size)
# Initialize storage
processed_images = []
processed_labels = []

# Process each image in the directory
for file_name in os.listdir(image_dir):
    if file_name.endswith('.png'):  # Only process PNG files
        file_path = os.path.join(image_dir, file_name)
        
        # Load the image
        image = cv2.imread(file_path)
        if image is None:
            print(f"Failed to load image: {file_path}")
            continue
        
        try:
            # Get image dimensions
            height, width = image.shape[:2]
            if width == 0 or height == 0 or width * height < cutoff:
                continue
            
            # Rotate if width > height
            if width > height:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                width, height = height, width
            
            # Resize to target size
            image = cv2.resize(image, output_size)
            
            # Normalize image to [-1, 1]
            image = (image / 127.5) - 1.0
            
            # Get label (assuming file_name maps to label index)
            label_index = int(file_name.split('.')[0]) - 1  # Adjust based on naming
            label = labels[label_index] if label_index < len(labels) else None
            if label is None:
                print(f"Label not found for image: {file_name}")
                continue
            
            # Store the processed image and label
            processed_images.append(image)
            processed_labels.append(label)
        
        except Exception as e:
            print(f"Error processing image {file_path}: {e}")
            continue

# Convert to numpy arrays
processed_images = np.array(processed_images, dtype=np.float32)
processed_labels = np.array(processed_labels, dtype=np.int32)

print(len(processed_images))
print(processed_labels.size)

# One-hot encode labels
n_classes = np.unique(processed_labels).size
print(n_classes)
processed_labels = to_categorical(processed_labels, num_classes=n_classes)
print(processed_labels)

# Save processed dataset for GAN training
np.savez_compressed('processed_dataset.npz', images=processed_images, labels=processed_labels)
print("Processed dataset saved to 'processed_dataset.npz'.")
