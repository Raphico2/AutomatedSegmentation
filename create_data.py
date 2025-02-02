from config import*
import cv2
import os
import numpy as np
from roifile import ImagejRoi
import matplotlib.pyplot as plt

def reorder_points(points):
    if len(points.shape) != 2 or points.shape[1] != 2:
        raise ValueError("Les points doivent être un tableau 2D de forme (n_points, 2).")
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    sorted_points = points[np.argsort(angles)]
    
    return sorted_points

def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Image non trouvée à l'emplacement {image_path}")
    
    if len(image.shape) == 2: 
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    return image

def load_roi(roi_path):
    roi = ImagejRoi.fromfile(roi_path)
    return roi

def draw_rois_on_image(image, roi_folder, key_word="heart", color='green'):
    if color == 'green':
        color_code = (0, 255, 0)
    elif color == 'red':
        color_code = (255, 0, 0)
    else: 
        color_code = (0, 0, 255)

    for filename in os.listdir(roi_folder):
        if filename.endswith(".roi"):
            roi_path = os.path.join(roi_folder, filename)
            
            if key_word in roi_path.lower(): 
                roi = load_roi(roi_path)

                coords = roi.coordinates()
                if len(coords) > 0:
                    # Réorganiser les points pour former un polygone correct
                    coords = reorder_points(coords)
                    
                    # Convertir en int32 et s'assurer que c'est en 3D
                    coords = np.array(coords, dtype=np.int32).reshape((-1, 1, 2))
                    
                    # Dessiner le polygone
                    cv2.polylines(image, [coords], isClosed=True, color=color_code, thickness=1)

    return image

def create_segmented_image(image_path, roi_folder, region_of_interest, color='green', visualize=True, output_image_folder=None, output_image_name=None):
    image = load_image(image_path)
    image_with_rois = draw_rois_on_image(image, roi_folder, key_word=region_of_interest, color=color)

    if visualize:
        plt.imshow(cv2.cvtColor(image_with_rois, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    
    if output_image_folder and output_image_name:
        output_image_path = os.path.join(output_image_folder, output_image_name)
        cv2.imwrite(output_image_path, image_with_rois)

def process_images(image_folder, roi_base_folder, output_folder, region_of_interest='heart', color='green'):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for image_filename in os.listdir(image_folder):
        if image_filename.endswith(".tif"):
            image_path = os.path.join(image_folder, image_filename)
            
            roi_folder = os.path.join(roi_base_folder, 'RoiSet'+ image_filename[8:11])

            output_image_name = "Segmented_" + image_filename
            
            create_segmented_image(
                image_path, 
                roi_folder, 
                region_of_interest,
                visualize=False, 
                color=color,
                output_image_folder=output_folder, 
                output_image_name=output_image_name
            )

if __name__ == "__main__":
    
    '''
    image_folder = 'imagesJTraining'
    roi_base_folder = 'ROItraining'
    output_folder = 'heart_segmented_dataset'
    process_images(image_folder, roi_base_folder, output_folder, 'heart', color='green')
    

    image_folder = 'imagesJTraining'
    roi_base_folder = 'ROItraining'
    output_folder = 'placenta_segmented_dataset'
    process_images(image_folder, roi_base_folder, output_folder, 'placenta', color='red')
    '''
    
    image_folder = 'imagesJTraining'
    roi_base_folder = 'ROItraining'
    output_folder = 'liver_segmented_dataset'
    process_images(image_folder, roi_base_folder, output_folder, 'liver', color='red')