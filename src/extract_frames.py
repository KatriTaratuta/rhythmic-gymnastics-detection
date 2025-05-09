import cv2
import os

def extract_frames(video_path, output_dir):
    # Открываем видео файл
    cap = cv2.VideoCapture(video_path)
    
    # Проверяем, открылся ли видеофайл
    if not cap.isOpened():
        print(f"Ошибка открытия видео: {video_path}")
        return
    
    # Получаем частоту кадров
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Частота кадров для {video_path}: {fps}")
    
    # Создаём выходную директорию для текущего видео
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Сохраняем кадр в файл
        frame_filename = os.path.join(output_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)
        
        frame_count += 1
    
    cap.release()
    print(f"Извлечено {frame_count} кадров из {video_path}.")

# Список путей к видео
video_paths = [
    "D:/rhythmic-gymnastics-detection/data/raw_videos/video1.mp4",  # Путь к первому видео
    "D:/rhythmic-gymnastics-detection/data/raw_videos/video2.mp4",  # Путь ко второму видео
    "D:/rhythmic-gymnastics-detection/data/raw_videos/video3.mp4"   # Путь к третьему видео
]

# Папка для сохранения кадров
output_dir = "outputs/frames_output"  # Папка для кадров

# Обрабатываем каждое видео
for video_path in video_paths:
    # Извлекаем имя видео без расширения для создания уникальной папки
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    
    # Создаём папку для текущего видео, если её нет
    if not os.path.exists(video_output_dir):
        os.makedirs(video_output_dir)
    
    # Извлекаем кадры для текущего видео
    extract_frames(video_path, video_output_dir)