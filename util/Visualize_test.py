from Visualize import visualize_image, visualize_batch, preprocessor
from Dataset import LandmarksDataset
from torch.utils.data import random_split
from DataPreprocessing import Preprocessor
import torch
import numpy as np
import torchvision.transforms.functional as TF

import sys
import os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)


train_images = LandmarksDataset(preprocessor, train = True)
test_images = LandmarksDataset(preprocessor, train = False)

'''image1, landmarks1 = train_images[64]
image1 = TF.to_pil_image(image1)
width, height = image1.size
image1 = TF.resize(image1, (200, 200))
image1 = np.array(image1)
image1 = TF.to_pil_image(image1)
image1 = TF.to_grayscale(image1)
image1 = TF.to_tensor(image1)
image1 = (image1 - image1.min())/(image1.max() - image1.min())
image1 = (2 * image1) - 1

landmarks1[0] *= 200 / width
landmarks1[1] *= 200 / height
landmarks1 = landmarks1 / 200
landmarks1 = landmarks1*2 - 1
visualize_image(image1, landmarks1)'''

'''image3, landmarks3 = train_images[65]
visualize_image(image3, landmarks3)

image4, landmarks4 = train_images[66]
visualize_image(image4, landmarks4)'''



len_val_set = int(0.1 * len(train_images))
len_train_set = len(train_images) - len_val_set


print(f'{len_train_set} images for training')
print(f'{len_val_set} images for validating')
print(f'{len(test_images)} images for testing')


train_images, val_images = random_split(train_images, [len_train_set, len_val_set])

batch_size = 32
train_data = torch.utils.data.DataLoader(train_images, batch_size = batch_size, shuffle = True)
val_data = torch.utils.data.DataLoader(val_images, batch_size = 2 * batch_size, shuffle = False)
test_data = torch.utils.data.DataLoader(test_images, batch_size = 2 * batch_size, shuffle = False)

for x, y in train_data:
    break

print(x.shape, y.shape, x.max(), x.min(), y.max(), y.min())

for x, y in val_data:
    break

print(x.shape, y.shape, x.max(), x.min(), y.max(), y.min())

for x, y in test_data:
    break

print(x.shape, y.shape, x.max(), x.min(), y.max(), y.min())


visualize_batch(x[:16], y[:16], shape = (4, 4), size = 16, title = 'Training Batch Samples')