from skimage import io
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import numpy as np
import sys
import os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)



class LandmarksDataset(Dataset):
    def __init__(self, preprocesor, train):
        self.root_dir = 'ivslab_facial_train/helen' if train == True else 'ivslab_facial_train/300W'

        self.image_paths = []
        self.landmarks = np.empty(shape = (0, 51, 2))
        self.preprocessor = preprocesor
        self.train = train

        if train != True:
            for i in range(1, len(os.listdir(os.path.join(self.root_dir, 'labels').replace("\\", "/"))) + 1):

                self.image_paths.append(os.path.join(self.root_dir, f'images/{"00" if i < 10 else "0" if i < 100 else ""}' + str(i) + '.png').replace("\\", "/"))

                with open(os.path.join(self.root_dir, f'labels/{"00" if i < 10 else "0" if i < 100 else ""}' + str(i) + '.pts').replace("\\", "/")) as file:
                    lines = file.readlines()

                landmark = []
                for line in lines[3:-1]:
                    coordinate = line.strip().split()
                    landmark.append([float(coordinate[0]), float(coordinate[1])])
                self.landmarks = np.concatenate((self.landmarks, [landmark]), axis = 0)

                self.landmarks = np.array(self.landmarks).astype('float32')
            assert len(os.listdir(os.path.join(self.root_dir, 'images').replace("\\", "/"))) == len(self.landmarks)
        else:
            image_path = os.listdir(os.path.join(self.root_dir, 'images').replace("\\", "/"))
            label_path = os.listdir(os.path.join(self.root_dir, 'labels').replace("\\", "/"))
            for i in range(len(os.listdir(os.path.join(self.root_dir, 'labels').replace("\\", "/")))):
                self.image_paths.append(self.root_dir + "/images/" + image_path[i])
                with open(self.root_dir + "/labels/" + label_path[i]) as file:
                    lines = file.readlines()

                landmark = []
                for line in lines[3:-1]:
                    coordinate = line.strip().split()
                    landmark.append([float(coordinate[0]), float(coordinate[1])])
                self.landmarks = np.concatenate((self.landmarks, [landmark]), axis = 0)
                self.landmarks = np.array(self.landmarks).astype('float32')

            # helen data sets seems to be insufficient, so I add another set of data in it
            image_path = os.listdir('ivslab_facial_train/IFPW/images')
            label_path = os.listdir('ivslab_facial_train/IFPW/labels')
            for j in range(len(os.listdir('ivslab_facial_train/IFPW/labels'))):
                self.image_paths.append('ivslab_facial_train/IFPW/images/' +  image_path[j])
                with open('ivslab_facial_train/IFPW/labels/' + label_path[j]) as file:
                    lines = file.readlines()
                landmark = []
                for line in lines[3:-1]:
                    coordinate = line.strip().split()
                    landmark.append([float(coordinate[0]), float(coordinate[1])])
                self.landmarks = np.concatenate((self.landmarks, [landmark]), axis =  0)
                self.landmarks = np.array(self.landmarks).astype('float32')
            assert len(os.listdir(os.path.join(self.root_dir, 'images').replace("\\", "/")) + os.listdir('ivslab_facial_train/IFPW/images')) == len(self.landmarks)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = io.imread(self.image_paths[index], as_gray = False)
        landmarks = self.landmarks[index].copy()

        image, landmarks = self.preprocessor(image, landmarks)

        return image, landmarks