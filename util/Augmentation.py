from torchvision import transforms
import numpy as np
import torchvision.transforms.functional as TF

import sys
import os

#
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

class FaceAugmentation:
    def __init__(self,
                 image_dim,
                 brightness,    
                 contrast,
                 saturation,
                 hue):
        
        self.image_dim = image_dim
        self.transform = transforms.ColorJitter(brightness, contrast, saturation, hue)
    
    def imageResize(self, image, landmarks):
        width, height = image.size
        image = TF.resize(image, (self.image_dim, self.image_dim))
        image = np.array(image)
        image = TF.to_pil_image(image)
        landmarks[:, 0] *= self.image_dim / width
        landmarks[:, 1] *= self.image_dim / height
        return image, landmarks
    
    def __call__(self, image, landmarks):
        #image = np.array(image)
        image, landmarks = self.imageResize(image, landmarks)
        #image = TF.to_pil_image(image)

        return self.transform(image), landmarks
    

class LandmarksAugmentation:
    def __init__(self, rotation_limit):
        self.rotation_limit = rotation_limit

    def random_rotation(self, image, landmarks):
        angle = np.random.uniform(-self.rotation_limit, self.rotation_limit)
        landmarks_transformation = np.array([
            [+np.cos(np.radians(angle)), -np.sin(np.radians(angle))], 
            [+np.sin(np.radians(angle)), +np.cos(np.radians(angle))]
        ])
        #landmarks = landmarks - 0.5
        image = TF.rotate(image, angle)
        transformed_landmarks = np.matmul(landmarks - 100, landmarks_transformation) + 100
        #transformed_landmarks = transformed_landmarks + 0.5

        return image, transformed_landmarks
    
    def __call__(self, image, landmarks):
        image, landmarks = self.random_rotation(image, landmarks)
        return image, landmarks