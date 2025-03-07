import os
import cv2
import numpy as np
import datetime
from retinaface import RetinaFace
from extractframes import extract_frames 
from crop import crop 
from facedetect import face_detect 
from PIL import Image 

video_path = "video.mp4"  # Change this to your actual video path
frames = extract_frames(video_path)
cropped_frames = crop(frames)

#display cropped frames as images
for frame in cropped_frames:
    img = Image.fromarray(frame)
    img.show()


#reassemble the cropped frames into a new video
output_video_path = "cropped_video.mp4"
height, width, _ = cropped_frames[0].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4

out = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))

for frame in cropped_frames:
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV
    out.write(frame_bgr)

out.release()
print("Cropped video saved as:", output_video_path)
