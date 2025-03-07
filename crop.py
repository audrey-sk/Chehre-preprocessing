from PIL import Image
import numpy as np

def crop(video):
    face_coords = face_detect(video)
    
    if not face_coords:
        return video  # No face detected, return original frames

    x, y, w, h = face_coords
    cropped_frames = []

    for frame in video:
        img = Image.fromarray(frame)  # Convert frame (NumPy) to PIL image
        
        # Crop the detected face region
        cropped = img.crop((x, y, x + w, y + h))
        
        # Resize while maintaining aspect ratio
        cropped = ImageOps.pad(cropped, (512, 512), color=(0, 0, 0))  # Black padding
        
        cropped_frames.append(np.array(cropped))  # Convert back to NumPy

    return cropped_frames