# Description: Detect faces in a video using OpenCV and Haar cascades.
import os
import cv2
import numpy as np

def detect_faces(video_path):
    # Load the pre-trained Haar cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get the original video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))  # Codec
    output_video = cv2.VideoWriter('output.mp4', fourcc, fps, (512, 512))  # Output video

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        
        # Get the cropped 512x512 frame
        cropped_frame = crop_frame(frame, faces)
        
        # Write to output video
        output_video.write(cropped_frame)

        # Display the cropped frame
        cv2.imshow('Cropped Face', cropped_frame)
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    output_video.release()
    cv2.destroyAllWindows()

def crop_frame(frame, faces, size=512):
    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2  # Default center of frame

    if len(faces) > 0:
        x, y, fw, fh = faces[0]  # Take the first detected face
        cx, cy = x + fw // 2, y + fh // 2  # Center the crop around the detected face

    # Ensure crop remains inside bounds
    x1 = max(cx - size // 2, 0)
    y1 = max(cy - size // 2, 0)
    x2 = min(x1 + size, w)
    y2 = min(y1 + size, h)

    cropped_frame = frame[y1:y2, x1:x2]

    # Resize to exactly 512x512 in case edges were cut off
    cropped_frame = cv2.resize(cropped_frame, (size, size), interpolation=cv2.INTER_LINEAR)

    return cropped_frame

# Example usage
detect_faces('video.mp4')