import os
from concurrent.futures.thread import ThreadPoolExecutor

import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from django.conf import settings
import time
import queue
from multiprocessing import Process, Queue

from threading import Event
import psutil

# Шлях до моделі
MODEL_PATH = os.path.join(settings.BASE_DIR, 'detection_app', 'best.pt')

# Завантаження моделі один раз при імпорті та переміщення на GPU, якщо доступний
model = YOLO(os.path.join(settings.BASE_DIR, 'detection_app', 'best.pt'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Налаштування параметрів для швидшої роботи
model.conf = 0.4  # поріг впевненості
model.iou = 0.5  # поріг IoU для NMS
model.max_det = 100  # максимальна кількість виявлень

print(f"YOLO модель ініціалізована на пристрої: {device}")


def detect_image(image_path):
    """Розпізнавання об'єктів на зображенні"""
    try:
        # Читаємо зображення
        img = cv2.imread(image_path)
        if img is None:
            raise Exception(f"Не вдалося прочитати зображення: {image_path}")

        # Змінюємо розмір для швидшої обробки, якщо зображення занадто велике
        h, w = img.shape[:2]
        if max(h, w) > 1280:
            scale = 1280 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        # 1. Детекція
        results = model(img)[0]  # беремо перший результат

        # 2. Візуалізація результатів
        result_img = results.plot()

        # 3. Збереження обробленого зображення
        result_dir = os.path.join(settings.MEDIA_ROOT, 'results')
        os.makedirs(result_dir, exist_ok=True)

        result_filename = f"result_{Path(image_path).stem}.jpg"
        result_path = os.path.join(result_dir, result_filename)
        cv2.imwrite(result_path, result_img)

        return result_path, results

    except Exception as e:
        print(f"❌ Помилка під час детекції зображення: {e}")
        return None, None


def detect_video(video_path):
    """Розпізнавання об'єктів на відео з використанням потоків"""
    try:
        # 1. Відкриваємо відео для читання
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Не вдалося відкрити відеофайл")

        # 2. Отримуємо параметри відео
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Перевіримо розмір відео і зменшимо його для швидшої обробки, якщо потрібно
        target_size = 640  # цільова висота або ширина для обробки
        resize_factor = 1.0
        if width > target_size or height > target_size:
            resize_factor = target_size / max(width, height)
            new_width = int(width * resize_factor)
            new_height = int(height * resize_factor)
            print(f"Зменшення розміру кадрів для обробки: {width}x{height} -> {new_width}x{new_height}")
        else:
            new_width, new_height = width, height

        # 3. Готуємо тимчасовий вихідний файл
        result_dir = os.path.join(settings.MEDIA_ROOT, 'results')
        os.makedirs(result_dir, exist_ok=True)
        output_filename = f"result_video_{Path(video_path).stem}_{int(time.time())}.mp4"
        output_path = os.path.join(result_dir, output_filename)

        # 4. Налаштовуємо відеозапис
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # або 'avc1' для h264
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 5. Створюємо черги та події для міжпотокової комунікації
        frame_queue = queue.Queue(maxsize=30)  # черга для зчитаних кадрів
        result_queue = queue.Queue(maxsize=30)  # черга для оброблених кадрів
        stop_event = Event()  # подія для зупинки потоків

        # Рахуємо кадри
        processed_frames = 0
        all_detections = []

        # 6. Функція для зчитування кадрів у окремому потоці
        def read_frames():
            nonlocal processed_frames
            frame_count = 0

            try:
                while cap.isOpened() and not stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1

                    # Пропускаємо кадри для прискорення (обробляємо кожен N-й кадр)
                    skip_factor = 3  # налаштуйте залежно від потреби
                    if frame_count % skip_factor != 0:
                        # Для пропущених кадрів зберігаємо оригінал
                        result_queue.put((frame, None, frame_count))
                        continue

                    # Зменшуємо розмір кадру для швидшої обробки
                    if resize_factor < 1.0:
                        resized_frame = cv2.resize(frame, (new_width, new_height))
                        frame_queue.put((resized_frame, frame, frame_count))
                    else:
                        frame_queue.put((frame, None, frame_count))

                    processed_frames += 1
            except Exception as e:
                print(f"Помилка в потоці зчитування кадрів: {e}")
            finally:
                # Сигналізуємо про завершення
                frame_queue.put((None, None, -1))
                print("Зчитування кадрів завершено")

        # 7. Функція для обробки кадрів у окремому потоці
        def process_frames():
            try:
                batch_size = 4  # розмір батчу для обробки
                batch_frames = []
                batch_originals = []
                batch_indices = []

                while not stop_event.is_set():
                    # Отримуємо кадр з черги
                    frame, original, frame_idx = frame_queue.get()

                    # Перевіряємо на сигнал завершення
                    if frame is None:
                        break

                    # Додаємо кадр до батчу
                    batch_frames.append(frame)
                    batch_originals.append(original if original is not None else frame)
                    batch_indices.append(frame_idx)

                    # Якщо зібрали батч або це останній кадр
                    if len(batch_frames) >= batch_size or frame_queue.qsize() == 0:
                        # Обробляємо батч разом
                        results = model(batch_frames, verbose=False)

                        # Обробляємо кожен результат
                        for i, (res, orig_frame, frame_idx) in enumerate(zip(results, batch_originals, batch_indices)):
                            # Візуалізуємо результати на оригінальному кадрі
                            result_frame = res.plot()

                            # Якщо був ресайз, повертаємо до оригінального розміру
                            if resize_factor < 1.0:
                                result_frame = cv2.resize(result_frame, (width, height))

                            # Збираємо інформацію про детекції
                            frame_detections = []
                            for box in res.boxes:
                                cls_id = int(box.cls.item())
                                conf = float(box.conf.item())
                                label = res.names[cls_id]

                                frame_detections.append({
                                    'label': label,
                                    'confidence': conf,
                                    'frame': frame_idx
                                })

                            # Додаємо до результатів
                            result_queue.put((result_frame, frame_detections, frame_idx))

                        # Очищаємо батч
                        batch_frames = []
                        batch_originals = []
                        batch_indices = []
            except Exception as e:
                print(f"Помилка в потоці обробки кадрів: {e}")
            finally:
                # Сигналізуємо про завершення
                result_queue.put((None, None, -1))
                print("Обробка кадрів завершено")

        # 8. Функція для запису результатів
        def write_results():
            nonlocal all_detections

            try:
                results_count = 0
                frames_buffer = {}  # буфер для збереження порядку кадрів
                next_frame_idx = 1  # наступний індекс кадру для запису

                progress_bar = tqdm(total=total_frames, desc="Обробка відео")

                while not stop_event.is_set():
                    # Отримуємо результат
                    result_frame, detections, frame_idx = result_queue.get()

                    # Перевіряємо на сигнал завершення
                    if result_frame is None and frame_idx == -1:
                        break

                    # Додаємо в буфер
                    frames_buffer[frame_idx] = (result_frame, detections)

                    # Записуємо кадри по порядку
                    while next_frame_idx in frames_buffer:
                        frame, dets = frames_buffer.pop(next_frame_idx)
                        out.write(frame)

                        if dets is not None:
                            all_detections.append(dets)

                        progress_bar.update(1)
                        next_frame_idx += 1
                        results_count += 1

                # Записуємо залишок кадрів
                for idx in sorted(frames_buffer.keys()):
                    frame, dets = frames_buffer[idx]
                    out.write(frame)

                    if dets is not None:
                        all_detections.append(dets)

                    progress_bar.update(1)
                    results_count += 1

                progress_bar.close()
                print(f"Записано {results_count} кадрів")

            except Exception as e:
                print(f"Помилка в потоці запису результатів: {e}")

        # 9. Запускаємо потоки через ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3) as executor:
            read_thread = executor.submit(read_frames)
            process_thread = executor.submit(process_frames)
            write_thread = executor.submit(write_results)

            try:
                # Чекаємо завершення всіх потоків
                read_thread.result()
                process_thread.result()
                write_thread.result()
            except Exception as e:
                print(f"Помилка при виконанні потоків: {e}")
                stop_event.set()  # зупиняємо всі потоки у випадку помилки

        # 10. Закриваємо ресурси
        cap.release()
        out.release()

        print(f"Обробку відео завершено! Знайдено {len(all_detections)} детекцій")
        return output_path, all_detections

    except Exception as e:
        print(f"❌ Помилка під час детекції відео: {e}")
        import traceback
        traceback.print_exc()
        return None, []


def detect_webcam_frame(frame):
    """Розпізнавання об'єктів на кадрі з веб-камери"""
    try:
        # Зменшуємо розмір для швидшого розпізнавання
        h, w = frame.shape[:2]
        if max(h, w) > 640:
            scale = 640 / max(h, w)
            resized_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            resized_frame = frame

        # 1. Детекція
        results = model(resized_frame)[0]

        # 2. Візуалізація результатів
        result_frame = results.plot()

        # Повертаємо до оригінального розміру, якщо змінювали
        if max(h, w) > 640:
            result_frame = cv2.resize(result_frame, (w, h))

        # 3. Збираємо інформацію про розпізнані об'єкти
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            label = results.names[cls_id]

            # Додаємо інформацію про координати боксу
            bbox = box.xyxy[0].tolist()  # конвертуємо до списку [x1, y1, x2, y2]

            detections.append({
                'label': label,
                'confidence': conf,
                'bbox': bbox
            })
        print(f"🔍 Використання CPU після обробки відео: {psutil.cpu_percent(interval=1)}%")
        return result_frame, detections

    except Exception as e:
        print(f"❌ Помилка під час детекції кадру веб-камери: {e}")
        # У випадку помилки повертаємо оригінальний кадр
        return frame, []