import cv2
import os

VIDEO_PATH = "F:\Adal\Descargas\Proyecto Vision Artificial\caso.mp4" 
OUTPUT_DIR = "F:/Adal/Descargas/vit_dataset/raw"

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
i = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if i % 10 == 0:  # cada 10 frames
        cv2.imwrite(f"{OUTPUT_DIR}/frame_{saved:05d}.jpg", frame)
        saved += 1

    i += 1

cap.release()
print("Frames extraídos:", saved)
