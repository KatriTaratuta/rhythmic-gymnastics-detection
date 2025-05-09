import os
import pandas as pd

# Определяем текущую директорию, где находится скрипт
current_dir = os.path.dirname(os.path.abspath(__file__))

# Папка с результатами обработки видео
output_folder = os.path.join(current_dir, '../outputs/')

# Путь для сохранения объединённого датасета
final_output_csv = os.path.join(current_dir, '../final_dataset.csv')

# Список для хранения всех DataFrame
all_dataframes = []

# Обрабатываем все CSV-файлы в папке с результатами
for filename in os.listdir(output_folder):
    if filename.endswith('_keypoints.csv'):
        csv_path = os.path.join(output_folder, filename)
        print(f'Чтение данных из {csv_path}')
        
        # Загружаем CSV в DataFrame
        df = pd.read_csv(csv_path)
        # Добавляем столбец с именем файла для идентификации источника
        df['video_filename'] = filename
        all_dataframes.append(df)

# Объединяем все DataFrame в один
final_df = pd.concat(all_dataframes, ignore_index=True)

# Сохраняем итоговый датасет в CSV
final_df.to_csv(final_output_csv, index=False)

print(f'Датасет успешно сохранён в {final_output_csv}')