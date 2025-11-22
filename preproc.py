import cv2

import numpy as np

def preprocess_base (frame, target_size=(640, 360)):
    #Rsized
    frame_resized= cv2.resize(frame, target_size)
    return frame_resized

def CLAHE(frame, clip_limit=2.0, tile_grid_size=(8, 8)):
    #enhanced contrast 
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    y_eq = clahe.apply(y)

    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    frame_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
    return frame_eq

def GAMMA(frame, gamma=1.3):
    #gamma < 1 light y gamma>1 dark
    frame_float = frame.astype(np.float32) / 255.0
    frame_gamma = np.power(frame_float, gamma)
    frame_out = np.clip(frame_gamma * 255.0, 0 , 255).astype(np.uint8)
    return frame_out

def denoise (frame):
    #Reduce noise for a better efficiency of the model 
    denoised = cv2.fastNlMeansDenoisingColored(
        frame, None,
        h=5,
        hColor=5,
        templateWindowSize=7,
        searchWindowSize=21
    )
    return denoised

def sharpen (frame):
    
    kernel = np.array([[0, 1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(frame, -1, kernel)
    return sharp

def medir_luma(frame):
    
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, _, _ = cv2.split(ycrcb)
    return float(y.mean())


def preprocess_frame (frame):
    
    #Size
    frame = preprocess_base(frame, (640, 360))


    #measure how bright is the video
    luma = medir_luma(frame)

    print(luma)

    if luma < 10:
        #Very Dark video 
        frame = CLAHE(frame, clip_limit=2.0, tile_grid_size=(8, 8))
        frame = GAMMA(frame, gamma=0.85)
        frame = sharpen(frame)
    elif luma <50:
        #Medium dark
        frame = CLAHE(frame, clip_limit=1.8, tile_grid_size=(8, 8))
        frame = GAMMA(frame, gamma=0.65)
        
    elif luma > 100:
        frame = CLAHE(frame, clip_limit=1.0, tile_grid_size=(16, 16))
    else:
        pass
         

    frame = denoise(frame)
    

    return frame
    

    


'''
video_path = "C:/Users/adalc/Downloads/proyecto vision artificial/videos estaciones de tren/18866687-hd_1080_1920_30fps.mp4"

while True:
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # fin del video

        frame_proc = preprocess_frame(frame)

        cv2.imshow("Original", frame)
        cv2.imshow("Preprocesado", frame_proc)

        # Si presionas ESC -> sale por completo
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            exit()
    
    cap.release()

    # Aquí espera a que presiones cualquier tecla para repetir el video
    print("Video terminado. Presiona cualquier tecla para repetir o ESC para salir...")
    key = cv2.waitKey(0) & 0xFF
    if key == 27:  # ESC para salir
        break

cv2.destroyAllWindows()

'''