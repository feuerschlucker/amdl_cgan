import os
import cv2
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt



def show_plot(images,labels, n):
    plt.figure(figsize=(13,6))
    for i in range(30):
        plt.subplot(3, 10, 1 + i)
        plt.axis('off')
        plt.title(f'{labels[i]}')
        plt.imshow(images[i, :, :, :])
    plt.tight_layout()
    plt.savefig('plots/random_test.png')






def main():
    
    image_dir0 = '/home/hroethl/amdl_cgan/data/train_cropped/images'
    label_file0= '/home/hroethl/amdl_cgan/data/train_cropped/train.mat'  

    image_dir1 = '/home/hroethl/amdl_cgan/data/test_cropped/images'
    label_file1 = '/home/hroethl/amdl_cgan/data/test_cropped/test.mat'  

    image_dirs=[image_dir0,image_dir1]
    label_files=[label_file0,label_file1]

    output_size = (28,50)  # Target size for images
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
            #print(file_name)
            file_path = os.path.join(image_dirs[batch], file_name)
            
            # Load the image
            image = cv2.imread(file_path)
            if image is None:
                print(f"Failed to load image: {file_path}")
                index= index+1
                continue
            
            try:
                # Get image dimensions
                height, width = image.shape[:2]
                if width == 0 or height == 0 or width * height < cutoff:
                    index= index+1
                    continue
                
                # Rotate if width > height
                if width > height:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    width, height = height, width
                
                # Resize to target size
                image = cv2.resize(image, output_size)
                
                # Normalize image to [-1, 1]
                image = (image / 127.5) - 1.0
                label = labels[index] 
                index= index+1
                
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
    print(n_classes)

    plt.hist(processed_labels)
    plt.savefig('plots/train_data_hist_labels.png')
    plt.close()



    no_images = len(processed_images)
    print(f'Size 2 batches  = {no_images}')
    index = np.random.randint(no_images, size=30)
    images = processed_images[index]
    labels= processed_labels[index]
    print(index)
    show_plot(images,labels,30)

    # Save processed dataset for GAN training
    # np.savez_compressed('processed_dataset_test.npz', images=processed_images, labels=processed_labels)
    print("Processed dataset saved to 'processed_dataset.npz'.")




if __name__ =='__main__':
     main()
