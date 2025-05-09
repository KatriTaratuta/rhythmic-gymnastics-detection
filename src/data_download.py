import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Класс для загрузки и предобработки данных
class PoseEstimationDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Загрузка изображения и метки
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        # Чтение изображения с помощью OpenCV
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Преобразуем в RGB

        if self.transform:
            image = self.transform(image)

        return image, label

# Трансформации для изображений
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Изменение размера до 224x224
    transforms.ToTensor(),  # Преобразование в тензор
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализация
])

# Загружаем данные
train_dataset = PoseEstimationDataset(csv_file='../outputs/final_dataset.csv', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)