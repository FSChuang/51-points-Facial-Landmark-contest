from .Augmentation import FaceAugmentation, LandmarksAugmentation
import numpy as np
import torchvision.transforms.functional as TF
import torch
from skimage import io
from matplotlib import pyplot as plt
from sklearn import preprocessing

import sys
import os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

minmax = preprocessing.MinMaxScaler()

class Preprocessor:
    def __init__(self,
                 image_dim,
                 brightness,
                 contrast,
                 saturation,
                 hue,
                 angle):
        
        self.image_dim = image_dim

        self.landmarks_augmentation = LandmarksAugmentation(angle)

        self.face_augmentation = FaceAugmentation(image_dim, brightness, contrast, saturation, hue)
    
    def __call__(self, image, landmarks):
        image = TF.to_pil_image(image)

        image, landmarks = self.face_augmentation(image, landmarks)

        image, landmarks = self.landmarks_augmentation(image, landmarks)

        landmarks = landmarks / np.array([*image.size])
        #landmarks = minmax.fit_transform(landmarks)
        landmarks = landmarks*2 - 1

        image = TF.to_grayscale(image)

        image = TF.to_tensor(image)

        image = (image - image.min())/(image.max() - image.min())
        image = (2 * image) - 1

        return image, torch.FloatTensor(landmarks.reshape(-1))