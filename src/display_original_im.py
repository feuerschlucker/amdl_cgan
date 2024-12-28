import json
import cv2
import os

# Paths to the JSON file and image directory
json_path = 'train.json'
image_dir = 'images'

# Load the JSON file
with open(json_path, 'r') as f:
    data = json.load(f)

# Create a mapping of category_id to category name
category_map = {cat['id']: cat['name'] for cat in data['categories']}

# Map image IDs to file names
image_map = {img['id']: img['file_name'] for img in data['images']}

# Extract bounding boxes and labels for each image
annotations_map = {}
for annotation in data['annotations']:
    image_id = annotation['image_id']
    bbox = annotation['bbox']
    category_id = annotation['category_id']
    label = category_map[category_id]
    
    if image_id not in annotations_map:
        annotations_map[image_id] = []
    annotations_map[image_id].append({'bbox': bbox, 'label': label})

# Function to draw bounding boxes on an image
def draw_bounding_boxes(image_path, annotations):
    image = cv2.imread(image_path)
    for annotation in annotations:
        bbox = annotation['bbox']
        label = annotation['label']
        
        # Convert bbox from [x, y, width, height] to [x1, y1, x2, y2]
        x1, y1, width, height = bbox
        x2, y2 = x1 + width, y1 + height
        
        # Draw the rectangle
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Put the label text above the rectangle
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

# Example: Draw bounding boxes for a specific image ID
image_id_to_view = 1  # Change this to the desired image ID
if image_id_to_view in annotations_map:
    image_file = image_map[image_id_to_view]
    image_path = os.path.join(image_dir, image_file)
    
    # Draw bounding boxes
    annotated_image = draw_bounding_boxes(image_path, annotations_map[image_id_to_view])
    
    # Display the image
    cv2.imshow('Annotated Image', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"No annotations found for image ID {image_id_to_view}.")
