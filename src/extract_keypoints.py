import cv2
import mediapipe as mp
import os
import pandas as pd

# Определяем текущую директорию, где находится скрипт
current_dir = os.path.dirname(os.path.abspath(__file__))

# Путь к папке с исходными видео и папке для выходных данных
input_folder = os.path.join(current_dir, '../data/raw_videos/')
output_folder = os.path.join(current_dir, '../outputs/')

# Создаём папку для выходных данных, если её нет
os.makedirs(output_folder, exist_ok=True)

print(f'Путь к папке с видео: {input_folder}')
print(f'Путь к папке с результатами: {output_folder}')

# Инициализация MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Функция для обработки видео и сохранения ключевых точек
def process_video(video_path, output_csv):
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
            keypoints_list.append(keypoints)

    cap.release()

    # Сохраняем результаты в CSV
    df = pd.DataFrame(keypoints_list)
    df.to_csv(output_csv, index=False)

# Обрабатываем все видео в папке
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.mp4', '.mov', '.avi')):
        video_path = os.path.join(input_folder, filename)
        output_csv = os.path.join(output_folder, f'{filename}_keypoints.csv')
        print(f'Обрабатываем: {video_path}')
        process_video(video_path, output_csv)
        print(f'Обработка завершена для {video_path}. Результаты сохранены в {output_csv}.')

print('Все видео обработаны.')

# Автоматический запуск подготовки данных после обработки видео
print('Запуск подготовки датасета...')
os.system('python prepare_dataset.py')