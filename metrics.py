import cv2
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
import numpy as np


def preprocess(frame):
    
    return frame


def main():
    
    model = YOLO("bestperson.pt")

    metrics = model.val(data="F:\Adal\Descargas\Proyecto Vision Artificial\Person detection.v16i.yolov8\data.yaml")


    print("Precision:", metrics.box.mp)      
    print("Recall:   ", metrics.box.mr)      
    print("mAP50:    ", metrics.box.map50)   
    print("mAP50-95: ", metrics.box.map)     


if __name__ == "__main__":
    main()
