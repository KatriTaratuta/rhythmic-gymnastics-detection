import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, frames_dir, keypoints_dir, transform=None):
        self.frames_dir = frames_dir
        self.keypoints_dir = keypoints_dir
        self.transform = transform
        self.video_names = os.listdir(frames_dir)

    def __len__(self):
        return len(self.video_names)
    
    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        
        # Загрузим кадры
        frames_path = os.path.join(self.frames_dir, video_name)
        frame_files = sorted(os.listdir(frames_path))  # Сортируем по имени для правильной последовательности
        
        frames = []
        for frame_file in frame_files:
            frame_path = os.path.join(frames_path, frame_file)
            frame = Image.open(frame_path).convert('RGB')
            frames.append(frame)
        
        # Загрузим ключевые точки
        base_name = os.path.splitext(video_name)[0]
        possible_keys = [f"{base_name}.MOV_keypoints.csv", f"{base_name}.mov_keypoints.csv"]

        keypoints_path = None
        for key in possible_keys:
            temp_path = os.path.join(self.keypoints_dir, key)
            if os.path.exists(temp_path):
                keypoints_path = temp_path
                break

        if not keypoints_path:
            raise FileNotFoundError(f"Keypoints file not found for {video_name}. Checked: {possible_keys}")

        keypoints = pd.read_csv(keypoints_path).values  # Чтение CSV
        
        # Если координаты нормализованы, умножим их на размеры кадра
        frame_width, frame_height = frames[0].size  # Получаем размеры первого кадра (ширина и высота)
        keypoints[:, 0] *= frame_width  # Масштабируем X-координаты
        keypoints[:, 1] *= frame_height  # Масштабируем Y-координаты
        
        # Преобразование в torch tensor
        frames = [np.array(frame) for frame in frames]
        frames = torch.tensor(np.stack(frames), dtype=torch.float32)  # [num_frames, H, W, C]
        keypoints = torch.tensor(keypoints, dtype=torch.float32)  # [num_frames, num_joints, 2]
        
        if self.transform:
            frames = self.transform(frames)
        
        return frames, keypoints

    def visualize_frame_with_keypoints(self, frame_idx=0):
        """Функция для визуализации кадра с ключевыми точками"""
        frames, keypoints = self[frame_idx]  # Получаем кадры и ключевые точки для конкретного индекса
        frame = frames[0].numpy().astype(np.uint8)  # Берем первый кадр
        
        plt.imshow(frame)  # Отображаем изображение
        
        # Проверим, что ключевые точки в пределах изображения
        for i in range(keypoints.shape[0]):  # По всем ключевым точкам
            x, y = keypoints[i, :2]  # Используем только первые два значения (x и y)
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:  # Убедимся, что точка внутри изображения
                plt.scatter(x, y, c='red', s=10)  # Наносим точку на картинку
        
        plt.title("Frame with Keypoints")
        plt.show()


# Пример использования CustomDataset
if __name__ == "__main__":
    frames_dir = 'D:/rhythmic-gymnastics-detection/outputs/frames'
    keypoints_dir = 'D:/rhythmic-gymnastics-detection/outputs'
    dataset = CustomDataset(frames_dir, keypoints_dir)
    
    # Проверим, сколько данных в датасете
    print(f"Total samples: {len(dataset)}")
    
    # Выведем информацию о первом примере
    frames, keypoints = dataset[0]
    print(f"Frames shape: {frames.shape}")
    print(f"Keypoints shape: {keypoints.shape}")

    # Визуализация первого кадра с ключевыми точками
    dataset.visualize_frame_with_keypoints(0)  # Визуализируем первый кадр