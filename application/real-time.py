import cv2
import numpy as np
import time
import torch
import dlib
import torchvision.transforms.functional as TF

import sys
import os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from model.model import XceptionNet

model = XceptionNet()
#model = model.cuda()
model.load_state_dict(torch.load('model.pt', map_location='cpu'))
model.eval()

LENGTH = 200

def preprocess_image(image):
    image = TF.to_pil_image(image)
    image = TF.resize(image, (200, 200))
    image = TF.to_tensor(image)
    image = (image - image.min())/(image.max() - image.min())
    image = (2 * image) - 1
    return image.unsqueeze(0)

@torch.no_grad()
def draw_landmarks_on_faces(frame, faces_landmarks):
    image = frame.copy()

    width, height, channels = image.shape
    landmarks = faces_landmarks.view(-1, 2)
    landmarks = (landmarks+1) * LENGTH * 0.5
    landmarks[:, 0] = landmarks[:, 0] * (height/LENGTH)
    landmarks[:, 1] = landmarks[:, 1] * (width/LENGTH)
    landmarks = landmarks.numpy()

    for (x, y) in enumerate(landmarks):
        try:
            cv2.circle(image, (int(y[0]), int(y[1])), 5, [40, 117, 255], -1)
        except:
            pass
        
    return image

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    prev_frame_time = 0
    new_frame_time = 0

    face_detector = dlib.get_frontal_face_detector()
    while True:
        ret, image = cap.read()
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if ret is None:
            break

        font = cv2.FONT_HERSHEY_SIMPLEX

        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))

        frame = preprocess_image(frame)
        landmarks_predictions = model(frame)
        image = draw_landmarks_on_faces(image, landmarks_predictions)

        cv2.putText(image, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow("Face", image)

        key = cv2.waitKey(25)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
