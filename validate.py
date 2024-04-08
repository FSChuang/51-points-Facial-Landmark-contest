import torch
from tqdm.auto import tqdm
import torch.optim as optim
import torch.nn as nn
import util.Visualize

from model.model import XceptionNet
from util.DataPreprocessing import Preprocessor
from util.Visualize import preprocessor, visualize_image, visualize_batch
from util.Dataset import LandmarksDataset

@torch.no_grad
def validate(model, val_data, save = None):
    objective = nn.MSELoss()
    cum_loss = 0.0

    model.eval()

    for features, labels in tqdm(val_data, desc = 'Validating', ncols = 600, bar_format='{l_bar}{bar:100}{r_bar}{bar:-100b}'):
        features = features.cuda()
        labels = labels.cuda()

        outputs = model(features)

        #loss = objective((outputs+1)*200*0.5, (labels+1)*200*0.5)
        loss = objective(outputs, labels)
        #loss = objective(outputs, (labels+1)*200*0.5)

        cum_loss += loss.item()

        break

    util.Visualize.visualize_batch(features[:16].cpu(), outputs[:16].cpu(), shape = (4, 4), size = 25, title = 'Validation', save = save)


    return cum_loss/(len(val_data)/20)

if __name__ == '__main__':
    model = XceptionNet()
    model = model.cuda()
    model.load_state_dict(torch.load('model.pt', map_location='cpu'))
    preprocessor = Preprocessor(
    image_dim = 200,
    brightness = 0.24,
    saturation = 0.3,
    contrast = 0.15,
    hue = 0.14,
    angle = 45)
    
    test_images = LandmarksDataset(preprocessor, train = False)
    test_data = torch.utils.data.DataLoader(test_images, batch_size = 2 * 25, shuffle = False)

    with torch.no_grad():
        for features, labels in tqdm(test_data, desc = 'Validating', ncols = 600, bar_format='{l_bar}{bar:100}{r_bar}{bar:-100b}'):
            features = features.cuda()
            labels = labels.cuda()

            outputs = model(features)

            break

        util.Visualize.visualize_batch(features[:25].cpu(), outputs[:25].cpu(), shape = (5, 5), size = 25, title = 'Validation', save = 'fuck_you.jpg')
