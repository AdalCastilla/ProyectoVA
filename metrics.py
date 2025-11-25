import cv2
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
import numpy as np


def preprocess(frame):
    # tu preprocesamiento
    return frame


def main():
    # 1. Cargar tu modelo base
    model = YOLO("bestperson.pt")

    metrics = model.val(data="F:\Adal\Descargas\Proyecto Vision Artificial\Person detection.v16i.yolov8\data.yaml")

# 3) imprime los valores importantes
    print("Precision:", metrics.box.mp)      # mean precision
    print("Recall:   ", metrics.box.mr)      # mean recall
    print("mAP50:    ", metrics.box.map50)   # mAP@0.5
    print("mAP50-95: ", metrics.box.map)     # mAP@[.5:.95]


if __name__ == "__main__":
    main()
