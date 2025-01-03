from uiutils import *

import cv2
from picamera2 import Picamera2

from ultralytics import YOLO

# Initialize the Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (320, 240)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()


# Load the exported NCNN model
ncnn_model = YOLO("yolo11n_ncnn_model")

while True:
    frame = picam2.capture_array()
    results = ncnn_model(frame)
    annotated_frame = results[0].plot()
    b, g, r = cv2.split(frame)
    img = cv2.merge((r, g, b))
    imgok = Image.fromarray(img)
    display.ShowImage(imgok)
    if button.press_b():
        break
