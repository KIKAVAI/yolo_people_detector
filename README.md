# YOLOv8 People Detector

Тестовое задание на Python, использующий модель YOLOv8n (nano) для детекции людей на видео с помощью OpenCV.

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/KIKAVAI/yolo_people_detector.git
cd yolo_people_detector
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Скачайте вес модели YOLOv8n:
```bash
# Вес можно скачать автоматически с помощью библиотеки Ultralytics:
# (это произойдёт при первом запуске)
```

4. Запустите скрипт:
```bash
python main.py
```

В процессе выполнения будет отображаться окно с результатами детекции, а видео с результатами будет сохранено в `output/detected_crowd.mp4`. Чтобы закрыть окно во время работы можно нажать **`q`**.

## Зависимости

- Python 3.8+
- OpenCV
- Ultralytics (YOLOv8)