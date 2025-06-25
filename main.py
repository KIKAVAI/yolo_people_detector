"""
YOLOv8 Video People Detector

Этот скрипт использует модель YOLOv8 для детекции людей на видео.
"""

import cv2
import os
from ultralytics import YOLO


def load_model(weights_path="yolov8n.pt"):
    """
    Загружает модель YOLOv8 по указанному пути.
    """
    return YOLO(weights_path)


def load_video(video_path):
    """
    Открывает видеофайл.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Не удалось открыть видео: {video_path}")
    return cap


def create_video_writer(output_path, width, height, fps):
    """
    Создаёт объект записи видео.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )


def draw_detections(frame, results, model):
    """
    Рисует рамки и подписи для обнаруженных объектов.
    """
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0].item())
            label = model.names[cls]

            if label == "person":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )


def main():
    """
    Основная функция: загрузка модели, обработка видео, сохранение результата.
    """
    model = load_model("yolov8n.pt")
    cap = load_video("videos/crowd.mp4")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = create_video_writer("output/detected_crowd.mp4", width, height, fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        draw_detections(frame, results, model)

        cv2.imshow("YOLOv8 - Детекция людей", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()