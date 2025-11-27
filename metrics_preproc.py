import cv2
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
import numpy as np


model = YOLO("yolov8n.pt")


val_images = Path("F:/Adal/Descargas/Proyecto Vision Artificial/RilwayTrack2025.v1-railwytrackv1.yolov8/valid/images")
val_labels = Path("F:/Adal/Descargas/Proyecto Vision Artificial/RilwayTrack2025.v1-railwytrackv1.yolov8/valid/labels")


def preprocess(frame):
    
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return frame


TP = 0
FP = 0
FN = 0


for img_path in tqdm(sorted(val_images.glob("*.jpg"))):
    frame = cv2.imread(str(img_path))
    frame = preprocess(frame)

   
    results = model(frame)[0]

   
    label_path = val_labels / (img_path.stem + ".txt")
    true_boxes = np.loadtxt(label_path, ndmin=2)

    
    true_boxes = true_boxes[true_boxes[:,0] == 0]

    det = [b for b in results.boxes if int(b.cls[0]) == 0]

    if len(det) > 0 and len(true_boxes) > 0:
        TP += 1
    elif len(det) > 0 and len(true_boxes) == 0:
        FP += 1
    elif len(det) == 0 and len(true_boxes) > 0:
        FN += 1


precision = TP / (TP + FP + 1e-6)
recall = TP / (TP + FN + 1e-6)

print("PRECISION:", precision)
print("RECALL:", recall)
