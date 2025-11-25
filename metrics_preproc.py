import cv2
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
import numpy as np

# 1. Cargar tu modelo base
model = YOLO("yolov8n.pt")

# 2. Carpeta del dataset de validación (el mismo de rail o personas)
val_images = Path("F:/Adal/Descargas/Proyecto Vision Artificial/RilwayTrack2025.v1-railwytrackv1.yolov8/valid/images")
val_labels = Path("F:/Adal/Descargas/Proyecto Vision Artificial/RilwayTrack2025.v1-railwytrackv1.yolov8/valid/labels")

# 3. Función de tu preprocesamiento
def preprocess(frame):
    # ejemplo: tu CLAHE + normalización
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return frame

# 4. Contadores para métricas
TP = 0
FP = 0
FN = 0

# 5. Procesar imágenes de validación
for img_path in tqdm(sorted(val_images.glob("*.jpg"))):
    frame = cv2.imread(str(img_path))
    frame = preprocess(frame)

    # correr YOLO
    results = model(frame)[0]

    # cargar labels reales
    label_path = val_labels / (img_path.stem + ".txt")
    true_boxes = np.loadtxt(label_path, ndmin=2)

    # filtrar solo clase persona (class 0)
    true_boxes = true_boxes[true_boxes[:,0] == 0]

    det = [b for b in results.boxes if int(b.cls[0]) == 0]

    if len(det) > 0 and len(true_boxes) > 0:
        TP += 1
    elif len(det) > 0 and len(true_boxes) == 0:
        FP += 1
    elif len(det) == 0 and len(true_boxes) > 0:
        FN += 1

# 6. Métricas
precision = TP / (TP + FP + 1e-6)
recall = TP / (TP + FN + 1e-6)

print("PRECISION:", precision)
print("RECALL:", recall)
