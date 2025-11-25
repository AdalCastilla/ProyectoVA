from ultralytics import YOLO
import cv2

model_base = YOLO("yolov8n.pt")

"F:\Adal\Descargas\Proyecto Vision Artificial\caso_recortado.mp4"
VIDEO_PATH = "F:\Adal\Descargas\Proyecto Vision Artificial\caso_recortado.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model_base(frame, conf=0.25)[0]

   
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls != 0:   
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"person {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Experimento 1 - YOLO base sin preproc", frame)
    if cv2.waitKey(20) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()