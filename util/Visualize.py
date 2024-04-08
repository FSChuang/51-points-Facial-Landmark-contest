from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from .DataPreprocessing import Preprocessor
import numpy as np
import time
import sys
import os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

preprocessor = Preprocessor(
    image_dim = 200,
    brightness = 0.5,
    saturation = 0.3,
    contrast = 0.15,
    hue = 0.14, 
    angle = 45)

def visualize_image(image, landmarks):

    plt.figure(figsize = (11, 11))
    image = (image - image.min())/(image.max() - image.min())

    landmarks = landmarks.view(-1, 2)
    landmarks = (landmarks + 1) * preprocessor.image_dim * 0.5

    plt.imshow(image[0], cmap = 'gray')
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s = 25, c = 'dodgerblue')
    plt.axis('off')
    plt.show(block = False)
    plt.pause(5)
    plt.close('all')
    #plt.show()
    
def visualize_batch(images_list, landmarks_list, size = 14, shape = (6, 6), title = None, save = None):
    fig = plt.figure(figsize = (size, size))
    grid = ImageGrid(fig, 111, nrows_ncols = shape, axes_pad = 0.08)
    for ax, image, landmarks in zip(grid, images_list, landmarks_list):
        image = (image - image.min())/(image.max() - image.min())

        landmarks = landmarks.view(-1, 2)
        landmarks = (landmarks + 1) * preprocessor.image_dim * 0.5
        #landmarks = (landmarks + 0.5) * preprocessor.image_dim
        landmarks = landmarks.numpy().tolist()
        landmarks = np.array([(x, y) for (x, y) in landmarks])

        ax.imshow(image[0], cmap = 'gray')
        ax.scatter(landmarks[:, 0], landmarks[:, 1], s = 10, c = 'dodgerblue')
        ax.axis('off')

    if title:
        print(title)
    if save:
        plt.savefig(save)
        
    plt.show(block = False)
    plt.pause(5)
    plt.close('all')
    #plt.show()

