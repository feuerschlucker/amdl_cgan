import os
import cv2
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from collections import Counter

def show_plot(images, labels, n):
    plt.figure(figsize=(13, 6))
    for i in range(30):
        plt.subplot(3, 10, 1 + i)
        plt.axis('off')
        plt.title(f'{labels[i]}')
        plt.imshow(cv2.transpose(images[i, :, :, :] + 1) / 2.0) 
    plt.tight_layout()
    plt.savefig('plots/random_test.png')

def balance_classes(images, labels):
    """
    Balances the classes in the dataset by oversampling underrepresented classes
    until all classes match the size of the largest class.
    """
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    max_count = max(label_counts)  # Largest class size
    balanced_images = []
    balanced_labels = []

    print(f"Class distribution before balancing: {dict(zip(unique_labels, label_counts))}")
    for label in unique_labels:
        # Get all images and labels of the current class
        class_indices = np.where(labels == label)[0]
        class_images = images[class_indices]
        class_labels = labels[class_indices]

        # Calculate how many more samples are needed
        num_to_add = max_count - len(class_images)

        # Oversample the class by randomly duplicating existing samples
        if num_to_add > 0:
            oversampled_indices = np.random.choice(len(class_images), size=num_to_add, replace=True)
            oversampled_images = class_images[oversampled_indices]
            oversampled_labels = class_labels[oversampled_indices]
            class_images = np.concatenate((class_images, oversampled_images), axis=0)
            class_labels = np.concatenate((class_labels, oversampled_labels), axis=0)

        # Add the balanced class data to the final dataset
        balanced_images.append(class_images)
        balanced_labels.append(class_labels)

    # Combine all balanced classes into a single dataset
    balanced_images = np.vstack(balanced_images)
    balanced_labels = np.concatenate(balanced_labels)

    print(f"Class distribution after balancing: {Counter(balanced_labels)}")
    return balanced_images, balanced_labels

def main():
    image_dir0 = '/home/hroethl/amdl_cgan/data/train_cropped/images'
    label_file0 = '/home/hroethl/amdl_cgan/data/train_cropped/train.mat'  

    image_dir1 = '/home/hroethl/amdl_cgan/data/test_cropped/images'
    label_file1 = '/home/hroethl/amdl_cgan/data/test_cropped/test.mat'  

    image_dirs = [image_dir0, image_dir1]
    label_files = [label_file0, label_file1]
    
    
    image_dirs = [image_dir0]
    label_files = [label_file0]

    output_size = (28, 50)  # Target size for images
    cutoff = 800  # Minimum pixel area
    processed_images = []
    processed_labels = []

    for batch in range(len(image_dirs)):
        # Load labels from .mat file
        mat_data = sio.loadmat(label_files[batch])
        labels = mat_data['y'].flatten()  
        file_nos = mat_data['names'].flatten()

        print(f'Size batch: {batch} = {len(labels)}')
        index = 0
        for file_no in file_nos:
            file_name = f'{file_no}.png'
            file_path = os.path.join(image_dirs[batch], file_name)
            
            # Load the image
            image = cv2.imread(file_path)
            if image is None:
                print(f"Failed to load image: {file_path}")
                index += 1
                continue
            
            try:
                # Get image dimensions
                height, width = image.shape[:2]
                if width == 0 or height == 0 or width * height < cutoff:
                    index += 1
                    continue
                
                # Rotate if width > height
                if width > height:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    width, height = height, width
                
                # Resize to target size
                image = cv2.resize(image, output_size)
                image = cv2.transpose(image)
                
                # Normalize image to [-1, 1]
                image = (image / 127.5) - 1.0
                label = labels[index] 
                index += 1
                
                # Store the processed image and label
                processed_images.append(image)
                processed_labels.append(label)
            
            except Exception as e:
                print(f"Error processing image {file_path}: {e}")
                break

    # Convert to numpy arrays
    processed_images = np.array(processed_images, dtype=np.float32)
    processed_labels = np.array(processed_labels, dtype=np.int32)

    n_classes = np.unique(processed_labels).size
    print(f"Number of classes: {n_classes}")

    plt.hist(processed_labels)
    plt.savefig('plots/train_data_hist_labels.png')
    plt.close()

    # Balance the dataset
    balanced_images, balanced_labels = balance_classes(processed_images, processed_labels)

    # Verify distribution
    plt.hist(balanced_labels)
    plt.savefig('plots/balanced_data_hist_labels.png')
    plt.close()

    # Randomly show samples
    no_images = len(balanced_images)
    print(f'Total balanced images = {no_images}')
    index = np.random.randint(no_images, size=30)
    images = balanced_images[index]
    labels = balanced_labels[index]
    show_plot(images, labels, 30)

    # Save balanced dataset for GAN training
    np.savez_compressed('balanced_dataset_train.npz', images=balanced_images, labels=balanced_labels)
    print("saved ")

if __name__ == '__main__':
    main()