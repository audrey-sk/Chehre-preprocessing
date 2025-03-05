import cv2
import numpy as np
import datetime
from retinaface import RetinaFace

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

def crop(video, point1x, point1y, width, height):  # Padding (ChatGPT) 1000x400
    return video[:, point1y:point1y + height, point1x:point1x + width]

def face_detect(video):  # Almost done "test.py"
    thresh = 0.8
    scales = [1024, 1980]
    gpuid = 0
    detector = RetinaFace('./model/R50', 0, gpuid, 'net3')
    
    # Assuming video is a numpy array (frame extraction required)
    img = video[0]  # Taking the first frame as example
    
    im_shape = img.shape
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    scales = [im_scale]
    flip = False
    
    faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)
    
    if faces is not None and faces.shape[0] > 0:
        box = faces[0].astype(int)  # Taking first detected face
        return box[0], box[1], box[2] - box[0], box[3] - box[1]
    
    return None, None, None, None

def chehre_pre_process(video):
	align(video)
	face_detect(video)
	video = crop(video, )