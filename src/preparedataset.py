import os
import cv2
import albumentations as A
import pandas as pd

# Папки для видео и результатов
input_folder = '../data/raw_videos/'
output_folder = '../outputs/'

# Папки для аугментированных данных
augmented_folder = os.path.join(output_folder, 'augmented')
frames_folder = os.path.join(output_folder, 'frames')

# Проверка наличия папок, создание при необходимости
os.makedirs(frames_folder, exist_ok=True)
os.makedirs(augmented_folder, exist_ok=True)

# Функция для извлечения кадров
def extract_frames(video_path, output_folder, frame_interval=30):
    print(f"Извлекаем кадры из {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    saved_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Сохраняем кадр через заданный интервал
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frames += 1
        frame_count += 1
    
    cap.release()
    print(f"Сохранено {saved_frames} кадров в {output_folder}")

# Функция для аугментации кадров
def augment_image(image):
    # Применяем аугментацию (поворот, отражение, изменение яркости/контрастности)
    transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomGamma(p=0.2),
    ])
    augmented = transform(image=image)
    return augmented["image"]

# Функция для аугментации и сохранения кадров
def augment_and_save_frames(input_folder, output_folder):
    print(f"Аугментируем кадры из {input_folder}")
    os.makedirs(output_folder, exist_ok=True)
    augmented_count = 0
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg"):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            
            # Аугментация
            augmented_image = augment_image(image)
            
            # Сохранение аугментированных изображений
            augmented_filename = os.path.join(output_folder, f"aug_{filename}")
            cv2.imwrite(augmented_filename, augmented_image)
            augmented_count += 1
            
    print(f"Сохранено {augmented_count} аугментированных кадров в {output_folder}")

# Создание датасета
def create_dataset(frames_folder, augmented_folder):
    print("Создание датасета...")
    data = []
    
    # Добавляем исходные кадры
    for video_folder in os.listdir(frames_folder):
        video_folder_path = os.path.join(frames_folder, video_folder)
        if os.path.isdir(video_folder_path):
            for frame_file in os.listdir(video_folder_path):
                if frame_file.endswith('.jpg'):
                    data.append([os.path.join(video_folder_path, frame_file), video_folder])
    
    # Добавляем аугментированные кадры
    for video_folder in os.listdir(augmented_folder):
        video_folder_path = os.path.join(augmented_folder, video_folder)
        if os.path.isdir(video_folder_path):
            for augmented_file in os.listdir(video_folder_path):
                if augmented_file.endswith('.jpg'):
                    data.append([os.path.join(video_folder_path, augmented_file), video_folder])

    # Сохраняем датасет в CSV
    df = pd.DataFrame(data, columns=['image_path', 'label'])
    df.to_csv(os.path.join(output_folder, 'final_dataset.csv'), index=False)
    print("Датасет сохранен в final_dataset.csv")

# Основной процесс
def main():
    # Поиск всех видеофайлов в папке raw_videos
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.mp4', '.mov', '.avi')):
            video_path = os.path.join(input_folder, filename)
            
            # Создание папок для кадров
            video_name = os.path.splitext(filename)[0]
            video_frames_folder = os.path.join(frames_folder, video_name)
            video_augmented_folder = os.path.join(augmented_folder, video_name)
            
            # Создаем папки для видео
            os.makedirs(video_frames_folder, exist_ok=True)
            os.makedirs(video_augmented_folder, exist_ok=True)
            
            # Извлекаем кадры
            extract_frames(video_path, video_frames_folder)
            
            # Аугментируем кадры
            augment_and_save_frames(video_frames_folder, video_augmented_folder)
    
    # Создание итогового датасета
    create_dataset(frames_folder, augmented_folder)

# Запуск основного процесса
if __name__ == "__main__":
    main()