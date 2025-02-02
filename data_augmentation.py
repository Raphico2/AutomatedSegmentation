import cv2
import numpy as np
import random
from config import*

def random_flip(image, roi):
    if random.random() > 0.5:
        image, roi = np.fliplr(image), np.fliplr(roi)
    if random.random() < 0.5:
        image, roi = np.flipud(image), np.flipud(roi)
    return image, roi

def random_rotate(image, roi, max_angle=MAX_ANGLE_ROTATION):
    angle = random.uniform(-max_angle, max_angle)
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, matrix, (w, h))
    rotated_roi = cv2.warpAffine(roi, matrix, (w, h))
    return rotated_image, rotated_roi

def random_zoom(image, roi, zoom_range=ZOOM_RANGE):
    zoom_factor = random.uniform(*zoom_range)
    h, w = image.shape[:2]
    new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
    zoomed_image = cv2.resize(image, (new_w, new_h))
    zoomed_roi = cv2.resize(roi, (new_w, new_h))

    if zoom_factor > 1.0:
        crop_h, crop_w = h, w
        start_x = (new_w - crop_w) // 2
        start_y = (new_h - crop_h) // 2
        cropped_image = zoomed_image[start_y:start_y+crop_h, start_x:start_x+crop_w]
        cropped_roi = zoomed_roi[start_y:start_y+crop_h, start_x:start_x+crop_w]
    else:
        pad_h, pad_w = (h - new_h) // 2, (w - new_w) // 2
        cropped_image = np.pad(zoomed_image, ((pad_h, h - new_h - pad_h), (pad_w, w - new_w - pad_w)), mode='constant')
        cropped_roi = np.pad(zoomed_roi, ((pad_h, h - new_h - pad_h), (pad_w, w - new_w - pad_w)), mode='constant')
    
    return cropped_image, cropped_roi

def random_shear(image, roi, shear_range=SHEAR_RANGE):
    shear_factor = random.uniform(*shear_range)
    h, w = image.shape[:2]
    matrix = np.array([[1, shear_factor, 0], [0, 1, 0]])
    sheared_image = cv2.warpAffine(image, matrix, (w, h))
    sheared_roi = cv2.warpAffine(roi, matrix, (w, h))
    return sheared_image, sheared_roi

def random_translate(image, roi, translate_range=TRANSLATE_RANGE):
    x_shift = random.uniform(*translate_range)
    y_shift = random.uniform(*translate_range)
    h, w = image.shape[:2]
    matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    translated_image = cv2.warpAffine(image, matrix, (w, h))
    translated_roi = cv2.warpAffine(roi, matrix, (w, h))
    return translated_image, translated_roi

def random_brightness(image, roi, brightness_range=BRIGHTNESS_RANGE):
    factor = random.uniform(*brightness_range)
    bright_image = cv2.convertScaleAbs(image, alpha=factor, beta=0)
    return bright_image, roi

def random_contrast(image, roi, contrast_range=CONTRAST_RANGE):
    factor = random.uniform(*contrast_range)
    mean = np.mean(image)
    contrast_image = cv2.convertScaleAbs(image, alpha=factor, beta=mean * (1 - factor))
    return contrast_image, roi

def random_gaussian_noise(image, roi, mean=MEAN_GAUSSIAN_NOISE, std_range=STD_GAUSSIAN_RANGE):
    std = random.uniform(*std_range)
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image, roi

# Fonction globale pour appliquer plusieurs augmentations de manière aléatoire
def apply_data_augmentation(image, roi):
    augmented_images = []
    augmented_rois = []

    # Appliquer une combinaison aléatoire de transformations
    for _ in range(N_AUGMENATIONS):  # 9 augmentations par image
        aug_image, aug_roi = image.copy(), roi.copy()
        
        if random.random() > 0.5:
            aug_image, aug_roi = random_flip(aug_image, aug_roi)
        if random.random() > 0.5:
            aug_image, aug_roi = random_rotate(aug_image, aug_roi)
        if random.random() > 0.5:
            aug_image, aug_roi = random_zoom(aug_image, aug_roi)
        if random.random() > 0.5:
            aug_image, aug_roi = random_shear(aug_image, aug_roi)
        if random.random() > 0.5:
            aug_image, aug_roi = random_translate(aug_image, aug_roi)
        if random.random() > 0.5:
            aug_image, aug_roi = random_brightness(aug_image, aug_roi)
        if random.random() > 0.5:
            aug_image, aug_roi = random_contrast(aug_image, aug_roi)
        if random.random() > 0.5:
            aug_image, aug_roi = random_gaussian_noise(aug_image, aug_roi)

        augmented_images.append(aug_image)
        augmented_rois.append(aug_roi)
    
    return augmented_images, augmented_rois
