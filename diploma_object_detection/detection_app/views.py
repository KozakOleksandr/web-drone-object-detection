from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .forms import CustomUserCreationForm
from django.contrib.auth import login, logout
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from pathlib import Path
import torch
import os
import base64
import cv2
import numpy as np
from ultralytics import YOLO
from .yolo import detect_image, detect_video, detect_webcam_frame


@login_required
def home_view(request):
    return render(request, 'detection_app/home.html')


@login_required
def video_view(request):
    return render(request, 'detection_app/video.html')


def register_view(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)  # автоматичний логін після реєстрації
            return redirect('home')  # заміни 'home' на свою головну сторінку
    else:
        form = CustomUserCreationForm()
    return render(request, 'detection_app/register.html', {'form': form})


def logout_view(request):
    logout(request)
    return redirect('login')


# Глобальна змінна для зберігання моделі
yolo_model = None

# Ініціалізація моделі YOLOv11
def initialize_model():
    try:
        # Шлях до вагів моделі
        model_path = os.path.join(settings.BASE_DIR, 'detection_app', 'best.pt')

        model = YOLO(model_path)
        # Переносимо модель на GPU, якщо доступна
        model.to('cuda' if torch.cuda.is_available() else 'cpu')

        # Налаштування параметрів для швидшої роботи
        model.conf = 0.3  # поріг впевненості
        model.iou = 0.5  # поріг IoU для NMS
        model.agnostic = False  # NMS для кожного класу окремо
        model.multi_label = False  # декілька міток для одного боксу
        model.max_det = 100  # максимальна кількість виявлень

        return model
    except Exception as e:
        print(f"Помилка при ініціалізації моделі: {e}")
        return None


@csrf_exempt
def detect_objects(request):
    """API для розпізнавання об'єктів на зображенні"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Метод не дозволений'}, status=405)

    try:
        # 1. Отримання зображення
        uploaded_file = request.FILES.get('image')
        if not uploaded_file:
            return JsonResponse({'error': 'Зображення не знайдено'}, status=400)
        if not uploaded_file.content_type.startswith('image'):
            return JsonResponse({'error': 'Завантажений файл не є зображенням'}, status=400)

        fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'uploads'))
        filename = fs.save(uploaded_file.name, uploaded_file)
        image_path = fs.path(filename)

        # 2. Детекція зображення через yolo.py
        result_path, results = detect_image(image_path)

        if result_path is None:
            return JsonResponse({'error': 'Помилка при обробці зображення'}, status=500)

        # 3. Формування відповіді з URL
        result_url = f"{settings.MEDIA_URL}results/{Path(result_path).name}"

        # 4. Збір метрик точності по класах
        class_metrics = {}
        detections = []

        # Обробка результатів детекції
        for box in results.boxes:
            # Отримуємо клас і точність для кожного боксу
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            label = results.names[cls_id]

            # Додаємо інформацію про виявлення
            detections.append({
                'label': label,
                'confidence': conf  # Перетворення у відсотки
            })

            # Оновлюємо максимальну точність для класу
            if label not in class_metrics or conf > class_metrics[label]:
                class_metrics[label] = conf

        # Перетворюємо точність у відсотки для відображення
        formatted_metrics = {cls: f"{cls}: {conf}%"
                             for cls, conf in class_metrics.items()}

        return JsonResponse({
            'image_url': result_url,
            'detections': detections,
            'class_metrics': formatted_metrics,
            'message': 'Розпізнавання завершено успішно'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': f'Помилка: {str(e)}'}, status=500)


@csrf_exempt
@login_required
def process_video(request):
    """API для розпізнавання об'єктів на відео"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Метод не дозволений'}, status=405)

    try:
        # Додамо більше діагностичної інформації
        print("Запит отримано:", request.FILES)

        # 1. Отримання відео
        uploaded_file = request.FILES.get('video')
        if not uploaded_file:
            print("Помилка: відео не знайдено в запиті")
            return JsonResponse({'error': 'Відео не знайдено'}, status=400)

        # Перевіряємо тип файлу більш гнучко
        content_type = uploaded_file.content_type
        print(f"Тип контенту: {content_type}")

        valid_video_types = ['video/mp4', 'video/webm', 'video/ogg', 'video/quicktime', 'video/x-msvideo']
        if not any(vtype in content_type.lower() for vtype in ['video']):
            print(f"Помилка: невірний тип файлу: {content_type}")
            return JsonResponse({'error': f'Завантажений файл не є відео. Тип: {content_type}'}, status=400)

        # 2. Збереження відео з перевіркою
        uploads_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)

        fs = FileSystemStorage(location=uploads_dir)
        try:
            filename = fs.save(uploaded_file.name, uploaded_file)
            video_path = fs.path(filename)
            print(f"Файл збережено: {video_path}")

            # Перевіряємо чи файл існує і читається
            if not os.path.exists(video_path):
                print("Помилка: файл не існує після збереження")
                return JsonResponse({'error': 'Помилка збереження відео'}, status=500)

            # Перевіряємо чи файл можна відкрити як відео
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("Помилка: не вдалося відкрити відео")
                return JsonResponse({'error': 'Формат відео не підтримується'}, status=400)
            cap.release()

        except Exception as e:
            print(f"Помилка при збереженні файлу: {e}")
            return JsonResponse({'error': f'Помилка при збереженні відео: {str(e)}'}, status=500)

        # 3. Детекція об'єктів на відео
        output_path, detections = detect_video(video_path)

        if output_path is None:
            print("Помилка: обробка відео не повернула результат")
            return JsonResponse({'error': 'Помилка при обробці відео'}, status=500)

        # 4. Формування відповіді з URL
        result_url = f"{settings.MEDIA_URL}results/{Path(output_path).name}"
        print(f"Успішно створено результат: {result_url}")

        # 5. Агрегуємо статистику детекцій
        aggregated_detections = {}
        for frame_detections in detections:
            for detection in frame_detections:
                label = detection['label']
                confidence = detection['confidence']

                if label not in aggregated_detections:
                    aggregated_detections[label] = {
                        'count': 0,
                        'total_confidence': 0,
                        'max_confidence': 0
                    }

                aggregated_detections[label]['count'] += 1
                aggregated_detections[label]['total_confidence'] += confidence
                aggregated_detections[label]['max_confidence'] = max(aggregated_detections[label]['max_confidence'],
                                                                     confidence)

        # Перетворення в формат для JSON-відповіді
        detections_list = []
        for label, stats in aggregated_detections.items():
            avg_confidence = stats['total_confidence'] / stats['count'] if stats['count'] > 0 else 0
            detections_list.append({
                'label': label,
                'count': stats['count'],
                'confidence': avg_confidence,
                'max_confidence': stats['max_confidence']
            })

        return JsonResponse({
            'video_url': result_url,
            'detections': detections_list,
            'message': 'Розпізнавання відео завершено успішно'
        })

    except Exception as e:
        import traceback
        print("Критична помилка при обробці відео:")
        traceback.print_exc()
        return JsonResponse({'error': f'Помилка: {str(e)}'}, status=500)


@csrf_exempt
@login_required
def process_webcam(request):
    """API для розпізнавання об'єктів на кадрі з веб-камери"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Метод не дозволений'}, status=405)

    try:
        # 1. Отримання кадру
        frame_file = request.FILES.get('frame')
        if not frame_file:
            return JsonResponse({'error': 'Кадр не знайдено'}, status=400)

        # 2. Читання зображення з байтів
        frame_bytes = frame_file.read()
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 3. Детекція об'єктів на кадрі
        result_frame, detections = detect_webcam_frame(frame)

        # 4. Конвертація результату в base64 для відправки на клієнт
        _, buffer = cv2.imencode('.jpg', result_frame)
        base64_image = base64.b64encode(buffer).decode('utf-8')

        return JsonResponse({
            'frame': base64_image,
            'detections': detections,
            'message': 'Кадр оброблено успішно'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': f'Помилка: {str(e)}'}, status=500)