from util.DataPreprocessing import Preprocessor
from util.Visualize import preprocessor, visualize_image, visualize_batch
from util.Dataset import LandmarksDataset
from model.model import XceptionNet
import cv2
import torchvision.transforms.functional as TF
import torch
import matplotlib.pyplot as plt
import matplotlib.image
from PIL import Image

LENGTH = 200

model = XceptionNet()
#model = model.cuda()
model.load_state_dict(torch.load('model_best_cos.pt', map_location='cpu'))
model.eval()

def preprocess_image(image):
    image = TF.to_pil_image(image)
    image = TF.resize(image, (200, 200))
    image = TF.to_tensor(image)
    image = (image - image.min())/(image.max() - image.min())
    image = (2 * image) - 1
    return image.unsqueeze(0)

def draw_landmarks_on_faces(frame, landmarks_predictions):
    image = frame.copy()

    width, height, depth = image.shape
    landmarks = landmarks_predictions.view(-1, 2)
    landmarks = (landmarks+1) * 200 * 0.5
    landmarks[:, 0] = landmarks[:, 0] * (height/LENGTH)
    landmarks[:, 1] = landmarks[:, 1] * (width/LENGTH)
    landmarks = landmarks.numpy()

    for (x, y) in enumerate(landmarks):
        try:
            cv2.circle(image, (int(y[0]), int(y[1])), 2, [40, 117, 255], -1)
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

if __name__ == '__main__':
    image = matplotlib.image.imread('application/image/Jobs.jpg')
    image2 = inference(image)
    #image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
    plt.imshow(image2)
    plt.show()
