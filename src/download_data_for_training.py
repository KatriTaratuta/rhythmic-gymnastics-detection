import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms
import numpy as np

class PoseEstimationDataset(Dataset):
    def __init__(self, frames_folder, keypoints_csv, transform=None):
        """
        frames_folder: путь к папке с изображениями
        keypoints_csv: CSV файл с ключевыми точками (позами)
        transform: преобразования для данных
        """
        self.frames_folder = frames_folder
        self.keypoints_df = pd.read_csv(keypoints_csv)  # Чтение ключевых точек
        self.transform = transform
        
    def __len__(self):
        return len(self.keypoints_df)
    
    def __getitem__(self, idx):
        # Загружаем кадр
        frame_path = f"{self.frames_folder}/frame_{idx}.jpg"
        frame = Image.open(frame_path).convert('RGB')
        
        # Получаем метки ключевых точек
        keypoints = self.keypoints_df.iloc[idx].values
        
        if self.transform:
            frame = self.transform(frame)
        
        return frame, torch.tensor(keypoints, dtype=torch.float32)

# Преобразования данных для входных изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Путь к данным
frames_folder = '../outputs/frames'
keypoints_csv = '../outputs/final_dataset.csv'

# Создание датасета
dataset = PoseEstimationDataset(frames_folder, keypoints_csv, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)