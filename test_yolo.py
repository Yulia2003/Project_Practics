import cv2
from ultralytics import YOLO
import numpy as np

# Путь к видеофайлу
video_path = "data/test.mp4"

# Загрузка модели YOLO
model = YOLO("yolov8n.pt")  # Используем предобученную модель YOLOv8 nano

def count_people_in_frame(frame):
    """Обрабатывает кадр и возвращает количество людей."""
    # Выполняем детекцию объектов на кадре (без трекинга)
    results = model(frame, conf=0.5)

    # Счетчик людей в текущем кадре
    people_count = 0

    # Минимальная площадь рамки для человека (в пикселях)
    min_area_threshold = 5000  # Можно настроить в зависимости от разрешения видео

    # Проходим по всем обнаруженным объектам
    for result in results:
        for box in result.boxes:
            # Проверяем, является ли объект человеком (класс 0 в COCO соответствует человеку)
            if int(box.cls) == 0:
                # Получаем координаты рамки
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Вычисляем площадь рамки
                area = (x2 - x1) * (y2 - y1)

                # Учитываем только рамки с площадью больше порога
                if area > min_area_threshold:
                    people_count += 1
                    # Рисуем рамку вокруг человека
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return people_count, frame

def process_video(video_path):
    """Обрабатывает видео, подсчитывает максимальное и общее количество людей."""
    # Открываем видеофайл
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видеофайл")
        return

    # Получаем параметры видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Создаем объект для записи выходного видео
    out = cv2.VideoWriter(
        'output_counted.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    max_people_in_frame = 0
    frame_count = 0
    unique_people_count = 0
    last_person_frame = -1  # Номер последнего кадра, где был человек
    gap_threshold = 50  # Количество кадров без людей, после которого считается новый человек
    person_intervals = []  # Список интервалов появления людей (для диагностики)
    current_interval_start = None  # Начало текущего интервала

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Подсчитываем людей в кадре
        people_count, annotated_frame = count_people_in_frame(frame)
        max_people_in_frame = max(max_people_in_frame, people_count)

        # Обновляем подсчет уникальных людей
        if people_count > 0:
            if current_interval_start is None:
                # Начало нового интервала
                current_interval_start = frame_count
            # Если человек появился после большого разрыва, считаем его новым
            if last_person_frame >= 0 and frame_count - last_person_frame > gap_threshold:
                unique_people_count += 1
                person_intervals.append((current_interval_start, frame_count))
                current_interval_start = frame_count
            last_person_frame = frame_count
        else:
            if current_interval_start is not None:
                # Завершаем текущий интервал, если были люди, а теперь их нет
                person_intervals.append((current_interval_start, frame_count))
                current_interval_start = None

        # Добавляем текст с количеством людей в кадре
        cv2.putText(
            annotated_frame,
            f'People in frame: {people_count}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        # Записываем кадр в выходное видео
        out.write(annotated_frame)

        # Отображаем кадр (можно закомментировать для ускорения обработки)
        cv2.imshow('People Counter', annotated_frame)

        frame_count += 1

        # Нажмите 'q' для выхода
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Финализируем подсчет уникальных людей
    if current_interval_start is not None:
        # Закрываем последний интервал
        person_intervals.append((current_interval_start, frame_count))
        unique_people_count += 1

    # Корректируем уникальных людей, если интервалов больше, чем подсчитано
    if len(person_intervals) > unique_people_count:
        unique_people_count = len(person_intervals)

    print(f"Максимальное количество людей в кадре: {max_people_in_frame}")
    print(f"Общее количество уникальных людей: {unique_people_count}")
    print(f"Обработано кадров: {frame_count}")
    print("Диагностика интервалов появления людей (начало, конец):")
    for i, (start, end) in enumerate(person_intervals, 1):
        print(f"Человек {i}: Кадры {start}–{end} (длительность: {end - start} кадров)")

    # Освобождаем ресурсы
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video(video_path)