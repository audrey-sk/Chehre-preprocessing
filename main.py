import cv2
import numpy as np
import datetime
from retinaface import RetinaFace

def chehre_pre_process(video):
	align(video)
	face_detect(video)
	video = crop(video, )
       


def align(video):  # 5 points, #StyleGAN
    # Align face to the center of the frame
    h, w, _ = video[0].shape  # Assuming first frame for alignment
    center_x, center_y = w // 2, h // 2
    
    p1x, p1y, width, height = face_detect(video)
    if p1x is not None:
        face_center_x = p1x + width // 2
        face_center_y = p1y + height // 2
        
        dx = center_x - face_center_x
        dy = center_y - face_center_y
        
        translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned_video = [cv2.warpAffine(frame, translation_matrix, (w, h)) for frame in video]
        return np.array(aligned_video)
    
    return video



