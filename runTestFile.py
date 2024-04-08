import sys
import os
import skimage.io
import io
import torch
import numpy as np
import torchvision.transforms.functional as TF
import cv2

from model.model import XceptionNet

test_dir = "ivslab_facial_test_private_qualification/"
result_landmarks_dir = 'qualification_result2'

model = XceptionNet(1)
model.load_state_dict(torch.load("model_copy2.pt"))

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = TF.to_pil_image(image)
    image = TF.resize(image, (200, 200))
    image = TF.to_tensor(image)
    image = (image - image.min())/(image.max() - image.min())
    image = (2 * image) - 1 

    return image.unsqueeze(0)

def landmarks2txt(landmarks, index):
    landmarks = landmarks.reshape(-1, 2)
    with open(os.path.join(result_landmarks_dir, f'image_{ "000" if index < 10 else "00" if index < 100 else "0" if index < 1000 else "" }' + str(index) + '.txt').replace("\\", "/"), "w") as file:
        file.write("version: 1\n")
        file.write("n_point: {}\n".format(len(landmarks)))
        file.write("{\n")

        for points in landmarks:
            file.write("{:.6f} {:.6f}\n".format(points[0], points[1]))

        file.write("}\n")

if __name__ == '__main__':

    if os.path.isdir('qualification_result2'):
        os.remove('qualification_result2')
    os.mkdir('qualification_result2')
    for index in range(1, len(os.listdir(test_dir))+1):
        image = skimage.io.imread(os.path.join(test_dir, f'image_{"000" if index < 10 else "00" if index < 100 else "0" if index < 1000 else ""}' + str(index) + '.png').replace("\\", "/"))
        image = preprocess_image(image)
        landmark_prediction = model(image)
        #landmark_prediction = (landmark_prediction + 1) * 200 * 0.5
        #landmark_prediction = landmark_prediction.detach().numpy()

        landmarks2txt(landmark_prediction, index)




    