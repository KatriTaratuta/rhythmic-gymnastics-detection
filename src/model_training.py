import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

# Инициализируем модель, оптимизатор и функцию потерь
model = HybridModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()  # Или MSELoss для регрессии

# Функция для тренировки модели
def train_model(model, train_loader, optimizer, criterion, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Обнуляем градиенты
            outputs = model(images)  # Прогоняем через модель

            loss = criterion(outputs, labels)  # Вычисляем потерю
            loss.backward()  # Вычисляем градиенты
            optimizer.step()  # Обновляем веса

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Обучение модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_model(model, train_loader, optimizer, criterion, num_epochs=10)