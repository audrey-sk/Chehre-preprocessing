import cv2
import numpy as np
from PIL import Image

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV) to RGB (PIL format)
        frames.append(frame)
    
    cap.release()
    return frames

video_path = "video.mp4"  # Change this to your actual video path
frames = extract_frames(video_path)
