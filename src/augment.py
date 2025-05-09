import cv2
import numpy as np
import albumentations as A
import os

def augment_image(image):
    # Применяем аугментацию (поворот, сдвиг, изменение яркости)
    transform = A.Compose([
        A.RandomRotate90(p=0.5),  # Случайный поворот на 90 градусов
        A.HorizontalFlip(p=0.5),   # Горизонтальный флип
        A.RandomBrightnessContrast(p=0.2),  # Случайное изменение яркости и контраста
        A.RandomGamma(p=0.2),      # Случайное изменение гаммы
    ])
    augmented = transform(image=image)
    return augmented["image"]

def augment_and_save_frames(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg"):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            
            # Аугментация
            augmented_image = augment_image(image)
            
            # Сохранение аугментированных изображений
            augmented_filename = os.path.join(output_folder, f"aug_{filename}")
            cv2.imwrite(augmented_filename, augmented_image)

# Применяем аугментацию к кадрам для каждого видео
augment_and_save_frames("output/video1_frames", "output/video1_augmented")
augment_and_save_frames("output/video2_frames", "output/video2_augmented")
augment_and_save_frames("output/video3_frames", "output/video3_augmented")