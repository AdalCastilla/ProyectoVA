from ultralytics import YOLO
import cv2
import numpy as np

from preproc import preprocess_frame

model_rails = YOLO("best(80).pt")
model_person = YOLO("best-human(50).pt")

VIDEO_PATH = "C:/Users/adalc/Downloads/proyecto vision artificial/videos estaciones de tren/caso_real.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_proc = preprocess_frame(frame)

    # Rieles: buscamos recall alto (puedes ajustar conf)
    results_rails = model_rails(frame_proc, conf=0.15)[0]
    rails_boxes = results_rails.boxes.xyxy.cpu().numpy() if len(results_rails.boxes) else []

    # Personas: bajamos conf para subir recall
    results_person = model_person(frame_proc, conf=0.10)[0]

    # Dibujar rieles
    for (rx1, ry1, rx2, ry2) in rails_boxes:
        rx1, ry1, rx2, ry2 = map(int, [rx1, ry1, rx2, ry2])
        cv2.rectangle(frame_proc, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
        cv2.putText(frame_proc, "RAILS", (rx1, ry1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    alerta = False

    for box in results_person.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Por defecto, persona verde
        color = (0,255,0)
        label = f"person {conf:.2f}"

        # Revisar si el centro cae en una caja de rieles
        for (rx1, ry1, rx2, ry2) in rails_boxes:
            rx1, ry1, rx2, ry2 = map(int, [rx1, ry1, rx2, ry2])
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                alerta = True
                color = (0,0,255)  # rojo
                label = "DANGER"
                break

        cv2.rectangle(frame_proc, (x1,y1), (x2,y2), color, 2)
        cv2.circle(frame_proc, (cx, cy), 4, color, -1)
        cv2.putText(frame_proc, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if alerta:
        cv2.putText(frame_proc, "ALERTA: PERSONA EN VIAS",
                    (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

    cv2.imshow("Experimento 3 - Sistema completo", frame_proc)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
