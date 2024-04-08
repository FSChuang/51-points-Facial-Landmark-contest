import cv2
import torch
import torchvision.transforms.functional as TF
from imutils import face_utils, resize
from moviepy.editor import VideoFileClip
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from sklearn import preprocessing
import dlib


import sys
import os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

LENGTH = 200

from model.model import XceptionNet

model = XceptionNet()
#model.to('cuda')
#state_dict = torch.load("model.pt")
model.load_state_dict(torch.load('model.pt', map_location='cpu'))
model.eval()

def preprocess_image(image):
    image = TF.to_pil_image(image)
    image = TF.resize(image, (200, 200))
    image = TF.to_tensor(image)
    image = (image - image.min())/(image.max() - image.min())
    image = (2 * image) - 1
    return image.unsqueeze(0)


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

@torch.no_grad()
def inference(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    #outputs = []
    preprocessed_image = preprocess_image(gray)
    landmarks_predictions = model(preprocessed_image)
    #outputs.append(landmarks_predictions.cpu())

    return draw_landmarks_on_faces(frame, landmarks_predictions)

face_detector = dlib.get_frontal_face_detector()

def output_video(video, name, seconds = None):
    total = int(video.fps * seconds) if seconds else int(video.fps * video.duration)
    print('Will read', total, 'images...')

    outputs = []

    writer = cv2.VideoWriter(name + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), video.fps, tuple(video.size))

    for i, frame in enumerate(tqdm(video.iter_frames(), total = total), 1):    
        if seconds:
            if (i + 1) == total:
                break

        faces = face_detector(frame, 1)

        if faces:     
            output = inference(frame)
        else:
            output = frame

        outputs.append(output)

        writer.write(cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

    writer.release()

    return outputs

if __name__ == '__main__':
    video = VideoFileClip("application/video/meme2.mp4")
    print('FPS: ', video.fps)
    print('Duration: ', video.duration, 'seconds')

    for frame in video.iter_frames():
        break

    
    outputs = output_video(video, "video_output/meme2 Face Detection")
    plt.figure(figsize = (11, 11))
    plt.imshow(outputs[10])

    plt.show()
    


