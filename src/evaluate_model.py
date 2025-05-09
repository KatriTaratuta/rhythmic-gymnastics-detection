from sklearn.metrics import accuracy_score, f1_score

# Функция для тестирования модели
def evaluate_model(model, test_loader):
    model.eval()  # Переводим модель в режим оценки
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)  # Прогоняем через модель
            preds = outputs.argmax(dim=1)  # Получаем предсказания

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Вычисляем метрики
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}")

# Оценка модели
evaluate_model(model, train_loader)  # Можно использовать тестовый DataLoader
