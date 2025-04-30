from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np
from ultralytics import YOLO
import os
import json
from datetime import datetime
import logging
import pandas as pd
import io

app = Flask(__name__)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Папка для загрузки
UPLOAD_FOLDER = 'Uploads'
HISTORY_FILE = 'history.json'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Загрузка модели YOLO
model = YOLO("yolov8n.pt")

def count_people_in_frame(frame):
    """Обрабатывает кадр и возвращает количество людей."""
    results = model(frame, conf=0.5)
    people_count = 0
    min_area_threshold = 5000

    for result in results:
        for box in result.boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > min_area_threshold:
                    people_count += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return people_count, frame

def process_video(video_path):
    """Обрабатывает видео, подсчитывает максимальное и общее количество людей."""
    logger.info(f"Начало обработки видео: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Не удалось открыть видеофайл")
        return {"error": "Не удалось открыть видеофайл"}

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    logger.info(f"FPS: {fps}")

    max_people_in_frame = 0
    frame_count = 0
    unique_people_count = 0
    last_person_frame = -1
    gap_threshold = 50
    person_intervals = []
    current_interval_start = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        people_count, annotated_frame = count_people_in_frame(frame)
        max_people_in_frame = max(max_people_in_frame, people_count)

        if people_count > 0:
            if current_interval_start is None:
                current_interval_start = frame_count
            if last_person_frame >= 0 and frame_count - last_person_frame > gap_threshold:
                unique_people_count += 1
                person_intervals.append((current_interval_start, frame_count))
                current_interval_start = frame_count
            last_person_frame = frame_count
        else:
            if current_interval_start is not None:
                person_intervals.append((current_interval_start, frame_count))
                current_interval_start = None

        frame_count += 1

    if current_interval_start is not None:
        person_intervals.append((current_interval_start, frame_count))
        unique_people_count += 1

    if len(person_intervals) > unique_people_count:
        unique_people_count = len(person_intervals)

    cap.release()
    logger.info(f"Обработка завершена: max_people={max_people_in_frame}, unique_people={unique_people_count}")

    return {
        "max_people": max_people_in_frame,
        "unique_people": unique_people_count,
        "frame_count": frame_count,
        "intervals": person_intervals,
        "status": "success"
    }

def process_image(image_path):
    """Обрабатывает изображение."""
    logger.info(f"Начало обработки изображения: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        logger.error("Не удалось открыть изображение")
        return {"error": "Не удалось открыть изображение"}

    people_count, annotated_frame = count_people_in_frame(frame)
    output_filename = f"output_{os.path.basename(image_path)}"
    output_path = os.path.join(UPLOAD_FOLDER, output_filename)
    cv2.imwrite(output_path, annotated_frame)

    return {
        "max_people": people_count,
        "unique_people": 1,
        "frame_count": 1,
        "intervals": [[0, 1]],
        "output_image": output_filename,
        "status": "success"
    }

def save_history(filename, results):
    """Сохраняет историю запросов в JSON."""
    history_entry = {
        "timestamp": datetime.now().isoformat(),
        "filename": filename,
        "max_people": results["max_people"],
        "unique_people": results["unique_people"],
        "frame_count": results["frame_count"],
        "intervals": results["intervals"]
    }
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
    history.append(history_entry)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)
    logger.info(f"История сохранена для {filename}")

def generate_excel_report():
    """Генерирует Excel-отчет из истории."""
    if not os.path.exists(HISTORY_FILE):
        logger.error("Файл истории не найден")
        return None

    try:
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
    except Exception as e:
        logger.error(f"Ошибка чтения history.json: {str(e)}")
        return None

    df = pd.DataFrame([
        {
            "Timestamp": entry["timestamp"],
            "Filename": entry.get("filename", entry.get("video_filename", "Unknown")),
            "Max People": entry["max_people"],
            "Unique People": entry["unique_people"],
            "Frame Count": entry["frame_count"],
            "Intervals": "; ".join([f"{start}-{end}" for start, end in entry["intervals"]])
        }
        for entry in history
    ])

    try:
        output = io.BytesIO()
        df.to_excel(output, index=False)
        output.seek(0)
        logger.info("Excel-отчет успешно сгенерирован")
        return output
    except Exception as e:
        logger.error(f"Ошибка генерации Excel: {str(e)}")
        return None

def generate_video_stream(video_path):
    """Генерирует поток кадров для стриминга из видео."""
    logger.info(f"Попытка открыть видео для стриминга: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Не удалось открыть видео для стриминга: {video_path}")
        return

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            logger.info(f"Конец видео или ошибка чтения: {video_path}")
            break

        people_count, annotated_frame = count_people_in_frame(frame)
        cv2.putText(
            annotated_frame,
            f'People in frame: {people_count}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ret:
            logger.error("Ошибка кодирования кадра в JPEG")
            continue

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    logger.info(f"Видеопоток завершен: {video_path}")

def generate_camera_stream():
    """Генерирует поток кадров с веб-камеры."""
    logger.info("Попытка открыть веб-камеру")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Не удалось открыть веб-камеру")
        return

    while True:
        success, frame = cap.read()
        if not success:
            logger.error("Ошибка чтения кадра с веб-камеры")
            break

        people_count, annotated_frame = count_people_in_frame(frame)
        cv2.putText(
            annotated_frame,
            f'People in frame: {people_count}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ret:
            logger.error("Ошибка кодирования кадра в JPEG")
            continue

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    logger.info("Поток с веб-камеры завершен")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        logger.error("Файл не загружен")
        return jsonify({"error": "Файл не загружен"}), 400

    file = request.files['file']
    if file.filename == '':
        logger.error("Файл не выбран")
        return jsonify({"error": "Файл не выбран"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    logger.info(f"Файл сохранен: {file_path}")

    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        results = process_image(file_path)
    else:
        results = process_video(file_path)

    if "error" in results:
        logger.error(f"Ошибка обработки: {results['error']}")
        return jsonify({"error": results["error"]}), 500

    save_history(file.filename, results)

    return jsonify({
        "max_people": results["max_people"],
        "unique_people": results["unique_people"],
        "frame_count": results["frame_count"],
        "intervals": results["intervals"],
        "filename": file.filename,
        "output_image": results.get("output_image"),
        "status": "success"
    })

@app.route('/video_feed')
def video_feed():
    filename = request.args.get('filename')
    if not filename:
        logger.error("Имя файла не указано для видеопотока")
        return jsonify({"error": "Файл не указан"}), 400

    video_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(video_path):
        logger.error(f"Видеофайл не найден: {video_path}")
        return jsonify({"error": "Файл не найден"}), 404

    logger.info(f"Запуск видеопотока для: {video_path}")
    return Response(generate_video_stream(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera_feed')
def camera_feed():
    logger.info("Запуск потока с веб-камеры")
    return Response(generate_camera_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/history')
def get_history():
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
            return jsonify(history)
        return jsonify([])
    except Exception as e:
        logger.error(f"Ошибка при загрузке истории: {str(e)}")
        return jsonify({"error": "Ошибка загрузки истории"}), 500

@app.route('/export_excel')
def export_excel():
    output = generate_excel_report()
    if output is None:
        logger.error("Не удалось сгенерировать Excel-отчет")
        return jsonify({"error": "Не удалось сгенерировать отчет"}), 500

    return Response(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment;filename=history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"}
    )

if __name__ == '__main__':
    app.run(debug=True)