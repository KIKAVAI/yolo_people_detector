"""
YOLOv8 Video People Detector

Этот скрипт использует модель YOLOv8 для детекции людей на видео.
"""

import cv2
import os
from ultralytics import YOLO

def main():
    """
    Основная функция для запуска детекции людей на видео.
    """
    # Загрузка модели YOLOv8
    model = YOLO('yolov8n.pt')

    # Загрузка видеофайла
    cap = cv2.VideoCapture('videos/crowd.mp4')

    # Проверка успешности открытия
    if not cap.isOpened():
        print('Error: Не удалось открыть видео.')
        return

    # Создание объекта записи видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    os.makedirs('output', exist_ok=True)
    out = cv2.VideoWriter('output/detected_crowd.mp4',
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (width, height))

    # Основной цикл обработки видео
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Получение результатов модели
        results = model(frame)

        # Обработка результатов модели
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                conf = box.conf[0]
                cls = int(box.cls[0].item())
                label = model.names[cls]

                if label == 'person':
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Отображение кадра
        cv2.imshow('YOLOv8 - Детекция людей', frame)
        out.write(frame) # Сохранение видео

        # Выход по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Очистка ресурсов модели
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
