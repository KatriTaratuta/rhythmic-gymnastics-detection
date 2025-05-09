import torch
import torch.nn as nn
import torchvision.models as models

# Простая модель трансформера
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, n_heads, num_layers, hidden_dim):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=n_heads, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        return self.transformer(x)

# Гибридная модель CNN + Transformer
class HybridModel(nn.Module):
    def __init__(self, transformer_input_dim=2048, n_heads=4, num_layers=4, hidden_dim=512):
        super(HybridModel, self).__init__()
        
        # Используем ResNet50 для извлечения признаков
        self.cnn = models.resnet50(pretrained=True)
        self.cnn.fc = nn.Identity()  # Убираем классификатор, оставляем только фичи
        
        # Трансформер для анализа временных зависимостей
        self.transformer = TransformerEncoder(transformer_input_dim, n_heads, num_layers, hidden_dim)
    
    def forward(self, x):
        # x: (batch_size, num_frames, C, H, W)
        batch_size, num_frames, C, H, W = x.shape
        
        # Применяем ResNet к каждому кадру
        cnn_features = []
        for i in range(num_frames):
            frame = x[:, i, :, :, :]
            feature = self.cnn(frame)  # Извлекаем признаки из каждого кадра
            cnn_features.append(feature)
        
        # Теперь у нас есть признаки для каждого кадра, конкатенируем их
        cnn_features = torch.stack(cnn_features, dim=1)  # (batch_size, num_frames, features_dim)
        
        # Применяем трансформер для анализа временных зависимостей
        transformer_output = self.transformer(cnn_features)  # (batch_size, num_frames, features_dim)
        
        return transformer_output

# Пример создания модели
model = HybridModel()

# Пример входа (batch_size=2, num_frames=5, channels=3, height=224, width=224)
input_data = torch.randn(2, 5, 3, 224, 224)  # Пример случайных данных
output = model(input_data)

print(output.shape)  # Должен быть (batch_size, num_frames, features_dim)