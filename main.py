import clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess = clip.load("ViT-B/32", device=device)


def dirty_or_clear(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model_clip.encode_image(image)
        text_features = model_clip.encode_text(clip.tokenize(["dirty car", "clear car"]).to(device))
    
        similarity = (image_features @ text_features.T).softmax(dim=-1)
        
        if similarity[0][0] > similarity[0][1]:
            print('грязная')
        else:
            print('чистая')

def broken_or_not(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model_clip.encode_image(image)
        text_features = model_clip.encode_text(clip.tokenize(["damaged car", "clear car"]).to(device))
    
        similarity = (image_features @ text_features.T).softmax(dim=-1)

        if similarity[0][0] > similarity[0][1]:
            print('повреждена')
        else:
            print('не повреждена')

from roboflow import Roboflow
rf = Roboflow(api_key="Djl97WAfilFh0HI0x63I")
project = rf.workspace("carpro").project("car-scratch-and-dent")
version = project.version(3)
dataset = version.download("yolov11")

from ultralytics import YOLO

model_yolo = YOLO("yolo11m-obb.pt")

model_yolo.train(
    data="Car-Scratch-and-Dent-3/data.yaml",
    epochs=0,             # количество эпох
    imgsz=640,               # размер изображений
    batch=16,                # размер батча
    patience=20,             # ранняя остановка
    workers=2,                # кол-во потоков (для Colab)
    device=0                  # GPU
)

import uuid
import time
import cv2
from ultralytics import YOLO

model_yolo = YOLO("/runs/obb/train/weights/best.pt")

def infer_yolo(image_path):
    start = time.time()
    
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить {image_path}")
    height, width = img.shape[:2]
    
    results = model_yolo.predict(image_path, conf=0.25, imgsz=640, verbose=False)
    preds = []

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        for box in r.boxes:
            cx, cy, w, h = box.xywh[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            preds.append({
                'x': float(cx),
                'y': float(cy),
                'width': float(w),
                'height': float(h),
                'confidence': conf,
                'class': cls_name,
                'class_id': cls_id,
                'detection_id': str(uuid.uuid4())
            })
    
    return {
        'inference_id': str(uuid.uuid4()),
        'time': time.time() - start,
        'image': {'width': width, 'height': height},
        'predictions': preds
    }

import cv2
import matplotlib.pyplot as plt

def show_img(image_path):
    print('ORIGINAL IMAGE')
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10,6))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def show_img_with_mask(image_path):
    print('\nBBOX + MASK IMAGE')
    
    result = infer_yolo(image_path)
    
    if len(result['predictions']) == 0:
        print("Объекты не найдены.")
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    vis = image.copy()

    for pred in result['predictions']:
        cx, cy = pred['x'], pred['y']
        w, h = pred['width'], pred['height']
        conf = pred['confidence']
        cls = pred['class']

        x1 = int(cx - w/2)
        y1 = int(cy - h/2)
        x2 = int(cx + w/2)
        y2 = int(cy + h/2)

        if cls.lower() == "scratch":
            overlay = vis.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), -1)  # закрашенный прямоугольник
            alpha = 0.3  # прозрачность
            vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)

        # рамка и текст для всех
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255,0,0), 2)
        cv2.putText(vis, f"{cls} {conf:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    plt.figure(figsize=(10,6))
    plt.imshow(vis)
    plt.axis('off')
    plt.show()

image_path = 'Car-Scratch-and-Dent-3/test/images/panel-damage-a-bad-dent-in-a-body-panel-on-a-ford-focus-car-DGENB1_jpg.rf.7851e62fcc0b566963c97a14bfd1c308.jpg'

torch.use_deterministic_algorithms(False)

show_img(image_path)

dirty_or_clear(image_path)
broken_or_not(image_path)

show_img_with_mask(image_path)