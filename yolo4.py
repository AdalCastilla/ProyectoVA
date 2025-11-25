from ultralytics import YOLO
import cv2
import torch
from preproc import preprocess_frame

# Modelos
model_person = YOLO("bestperson.pt").to("cuda")  
model_rails  = YOLO("best.pt").to("cuda")  

# VideoF:\Adal\Descargas\Proyecto Vision Artificial\caso_recortado.mp4
VIDEO_PATH = "F:\Adal\Descargas\Proyecto Vision Artificial\caso_recortado.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

def boxes_overlap(boxA, boxB, min_overlap=1):
    
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    # coordenadas de la intersección
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    # ancho y alto de la intersección
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter_area = iw * ih

    return inter_area >= min_overlap


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_proc = preprocess_frame(frame)

    
    results_rails = model_rails(frame_proc, conf=0.25, verbose=False)[0]
    rails_boxes = []
    for b in results_rails.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        rails_boxes.append((x1, y1, x2, y2))
        cv2.rectangle(frame_proc, (x1, y1), (x2, y2), (0,255,255), 2)
        cv2.putText(frame_proc, "rail", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    #Deteccion personas
    results_person = model_person(frame_proc, conf=0.25, verbose=False)[0]

    alerta = False

    for box in results_person.boxes:
        cls = int(box.cls[0])
        if cls != 0:  
            continue
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        
        
        person_box = (x1, y1, x2, y2)

        danger = False
        for (rx1, ry1, rx2, ry2) in rails_boxes:
            rail_box = (rx1, ry1, rx2, ry2)
            if boxes_overlap(person_box, rail_box, min_overlap=50):  
                danger = True
                alerta = True
                break

        
        
        color = (0,0,255) if danger else (0,255,0)
        label = "DANGER" if danger else f"person {conf:.2f}"

        cv2.rectangle(frame_proc, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame_proc, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


    # Alerta
    if alerta:
        cv2.putText(frame_proc, "ALERTA: PERSONA EN VIAS",
                    (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0,0,255), 3)

    # Mostrar
    cv2.imshow("Sistema - Personas + Rieles", frame_proc)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
