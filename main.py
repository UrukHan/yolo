import cv2
import json
from ultralytics import YOLO
import torch
from tqdm import tqdm

# Проверка GPU и FP16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
half_precision = device == 'cuda'
print(f'Используется устройство: {device}, FP16: {half_precision}')

# Загрузка конфигурации
with open('config.json', 'r') as f:
    cfg = json.load(f)

# Модель YOLO на GPU
model = YOLO(cfg["model_name"]).to(device)

# Источник видео
MODE = cfg["mode"]
SOURCE = cfg["source"]

cap = cv2.VideoCapture(SOURCE)

# Параметры исходного видео
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

resize_dims = (width // 2, height // 2)

if MODE == 'file':
    output_filename = SOURCE.rsplit('.', 1)[0] + '_output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_fps = 10
    writer = cv2.VideoWriter(output_filename, fourcc, output_fps, resize_dims)

def get_area(zone):
    return (
        int(zone["xa1"] * resize_dims[0]),
        int(zone["ya1"] * resize_dims[1]),
        int(zone["xa2"] * resize_dims[0]),
        int(zone["ya2"] * resize_dims[1])
    )

area1 = get_area(cfg["zones"]["area1"])
area2 = get_area(cfg["zones"]["area2"])

def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])

def intersection_area(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0
    return (x_right - x_left) * (y_bottom - y_top)

# Предварительно загружаем и уменьшаем кадры с отображением прогресса
frames = []
frame_counter = 0

print("Загрузка и изменение размера кадров...")

pbar = tqdm(unit="frame")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_counter += 1
    if frame_counter % 6 == 0:
        frame = cv2.resize(frame, resize_dims)
        frames.append(frame)
    pbar.update(1)
pbar.close()

cap.release()
print(f'✅ Кадров после обработки: {len(frames)}')

# Обрабатываем кадры батчами с корректным прогрессом
batch_size = 16
print("Обработка кадров на GPU...")
for i in tqdm(range(0, len(frames), batch_size)):
    batch = frames[i:i + batch_size]
    results_batch = model.predict(batch, classes=[2,3,5,7], device=device, half=half_precision, verbose=False)

    for frame, results in zip(batch, results_batch):
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)

        for box in boxes:
            intersect_area1 = intersection_area(box, area1)
            intersect_area2 = intersection_area(box, area2)

            color = tuple(cfg["zones"]["default_color"])

            if (intersect_area1 > intersect_area2 and
                intersect_area1 > cfg["zones"]["area1"]["intersection_threshold"] * box_area(area1)):
                color = tuple(cfg["zones"]["area1"]["color"])
            elif (intersect_area2 >= intersect_area1 and
                  intersect_area2 > cfg["zones"]["area2"]["intersection_threshold"] * box_area(area2)):
                color = tuple(cfg["zones"]["area2"]["color"])

            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

        # Визуализация зон (белый цвет)
        zone_color = (255, 255, 255)
        cv2.rectangle(frame, area1[:2], area1[2:], zone_color, 2)
        cv2.putText(frame, 'Zone 1', (area1[0] + 5, area1[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_color, 2)
        cv2.rectangle(frame, area2[:2], area2[2:], zone_color, 2)
        cv2.putText(frame, 'Zone 2', (area2[0] + 5, area2[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_color, 2)

        if MODE == 'file':
            writer.write(frame)
        else:
            cv2.imshow('Detection YOLO11', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if MODE == 'file':
    writer.release()
    print(f'✅ Видео успешно сохранено в {output_filename}')
else:
    cv2.destroyAllWindows()
