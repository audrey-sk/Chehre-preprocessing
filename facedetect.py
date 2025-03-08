from PIL import Image, ImageOps
import numpy as np
from retinaface import RetinaFace 

def face_detect(video):
    thresh = 0.8
    scales = [1024, 1980]
    gpuid = 0
    detector = RetinaFace('./model/R50', 0, gpuid, 'net3')

    # Assuming video is a list/array of frames
    first_frame = video[0]  # Taking the first frame for face detection
    im_shape = first_frame.shape
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)

    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    scales = [im_scale]
    flip = False

    faces, landmarks = detector.detect(first_frame, thresh, scales=scales, do_flip=flip)

    if faces is not None and faces.shape[0] > 0:
        box = faces[0].astype(int)  # Taking the first detected face
        return box[0], box[1], box[2] - box[0], box[3] - box[1]

    return None  # No face detected