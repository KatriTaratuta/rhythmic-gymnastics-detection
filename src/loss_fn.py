import torch
import torch.nn as nn

class MPJPE_Loss(nn.Module):
    def __init__(self):
        super(MPJPE_Loss, self).__init__()

    def forward(self, predictions, targets):
        """
        predictions: предсказания модели (формат [batch_size, num_joints, 2])
        targets: реальные метки (формат [batch_size, num_joints, 2])
        """
        return torch.mean(torch.norm(predictions - targets, dim=2))

# Пример использования
loss_fn = MPJPE_Loss()

# Пример случайных данных для предсказаний и целей (реальных меток)
batch_size = 2
num_joints = 17  # Количество суставов
predictions = torch.randn(batch_size, num_joints, 2)  # Предсказания модели (случайные данные)
targets = torch.randn(batch_size, num_joints, 2)  # Реальные метки (случайные данные)

# Вычисление потери
loss = loss_fn(predictions, targets)
print(f'Loss: {loss.item()}')