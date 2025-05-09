import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from ResNetandTransformers import HybridModel  # Импортируем HybridModel
from loss_fn import MPJPE_Loss  # Импортируем функцию потерь
from CustomDataset import CustomDataset   # Импортируем Dataset
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Инициализация потерь
loss_fn = MPJPE_Loss()

# Инициализация модели
model = HybridModel()

# Оптимизатор
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Путь к данным
frames_dir = 'D:/rhythmic-gymnastics-detection/outputs/frames'  # Правильный путь
keypoints_dir = 'D:/rhythmic-gymnastics-detection/outputs'

# Функция collate_fn для фиксированной длины последовательности
def collate_fn(batch):
    max_seq_len = 10

    inputs = []
    labels = []

    for inp, label in batch:
        if inp.size(0) > max_seq_len:
            inp = inp[:max_seq_len]
        elif inp.size(0) < max_seq_len:
            padding = torch.zeros(max_seq_len - inp.size(0), *inp.shape[1:])
            inp = torch.cat([inp, padding], dim=0)

        inp = inp.permute(0, 3, 1, 2)

        inputs.append(inp)

        if label.size(0) > max_seq_len:
            label = label[:max_seq_len]
        elif label.size(0) < max_seq_len:
            label_padding = torch.zeros(max_seq_len - label.size(0), *label.shape[1:])
            label = torch.cat([label, label_padding], dim=0)

        labels.append(label)

    inputs = torch.stack(inputs)
    labels = torch.stack(labels)

    return inputs, labels

# Создание датасета и DataLoader
train_dataset = CustomDataset(frames_dir, keypoints_dir)
dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# Тренировка модели
num_epochs = 50
best_loss = float('inf')
loss_log = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % 10 == 0 or (i + 1) == len(dataloader):
            print(f"Epoch {epoch+1}/{num_epochs}, Step {i+1}/{len(dataloader)}, Loss: {loss.item():.6f}")

    epoch_loss = running_loss / len(dataloader)
    loss_log.append(epoch_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {epoch_loss:.6f}")

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print("Model saved!")

# Сохранение логов потерь
with open('train_log.txt', 'w') as f:
    for loss in loss_log:
        f.write(f"{loss}\n")

# Построение графика потерь
plt.plot(loss_log)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()