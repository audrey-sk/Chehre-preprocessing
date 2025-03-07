from PIL import Image
import numpy as np

def crop(video, point1x, point1y, width, height):
    cropped_frames = []
    target_size = 512
    
    for frame in video:  # Assuming video is a list of PIL images
        img = frame.crop((point1x, point1y, point1x + width, point1y + height))
        
        # Ensure square size by padding if necessary
        new_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))  # Black padding
        img_w, img_h = img.size
        
        offset_x = (target_size - img_w) // 2
        offset_y = (target_size - img_h) // 2
        new_img.paste(img, (offset_x, offset_y))
        
        cropped_frames.append(new_img)
    
    return cropped_frames