import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from util.DataPreprocessing import Preprocessor
from util.Visualize import preprocessor, visualize_image, visualize_batch
from util.Dataset import LandmarksDataset
from model.model import XceptionNet
from validate import *
from util.plot import plot
import os

torch.cuda.empty_cache()

def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)  
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0.1)

preprocessor = Preprocessor(
    image_dim = 200,
    brightness = 0.24,
    saturation = 0.3,
    contrast = 0.15,
    hue = 0.14,
    angle = 30)

train_images = LandmarksDataset(preprocessor, train = True)
test_images = LandmarksDataset(preprocessor, train = False) #300W

len_val_set = int(0.1 * len(train_images))
len_train_set = len(train_images) - len_val_set
train_images, val_images = random_split(train_images, [len_train_set, len_val_set])

batch_size = 15
train_data = torch.utils.data.DataLoader(train_images, batch_size = batch_size, shuffle = True)
val_data = torch.utils.data.DataLoader(val_images, batch_size = 2 * batch_size, shuffle = False)
test_data = torch.utils.data.DataLoader(test_images, batch_size = 2 * batch_size, shuffle = False)

#import model
model = XceptionNet()
model.apply(weights_init_uniform_rule)
model = model.cuda()

# initializing the objective loss & optimizer
objective = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 4e-4)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor=0.1, patience=3, verbose=False, min_lr=0.0000001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 10, T_mult=2, eta_min=1e-8)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15, gamma = 0.1, last_epoch=-1, verbose = False)

#create directory for storing the progress(gif form)
if os.path.isdir('progress'):
    os.remove('progress')
os.mkdir('progress')

#Start training the model
epochs = 315
batches = len(train_data)
best_loss = np.inf
optimizer.zero_grad()

training_loss = []
validate_loss = []
lr_list = []


for epoch in range(epochs):
    cum_loss = 0.0

    lr_list.append(optimizer.param_groups[0]['lr'])
    plot(train_loss=lr_list, save = 'graph/learning_rate.jpg', line = 'learning rate')

    model.train()
    for batch_index, (features, labels) in enumerate(tqdm(train_data, desc = f'Epoch({epoch + 1}/{epochs})', ncols = 800, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')):
        features = features.cuda()
        labels = labels.cuda()

        outputs = model(features)
        '''if batch_index == 0:
            print(labels)
            print("---------------")
            print(outputs)
            print("===============")
            print((outputs+1)*preprocessor.image_dim*0.5)
            print("---------------")
            print((labels+1)*preprocessor.image_dim*0.5)'''

        #loss = objective((outputs+1)*preprocessor.image_dim*0.5, (labels+1)*preprocessor.image_dim*0.5)
        loss = objective(outputs, labels)
        #loss = objective((outputs + 0.5) * preprocessor.image_dim, (labels + 0.5) * preprocessor.image_dim)
        #loss = objective(outputs, (labels+1)*preprocessor.image_dim*0.5)

        loss.backward()

        #scheduler.step(loss)
        optimizer.step()

        optimizer.zero_grad()

        cum_loss += loss.item()
    
    scheduler.step()
    
    training_loss.append(cum_loss/batches)
    val_loss = validate(model, val_data, os.path.join('progress', f'epoch({str(epoch + 1).zfill(len(str(epochs)))}).jpg'))
    validate_loss.append(val_loss)
    plot(training_loss, validate_loss, 'graph/plot.jpg', 'loss')

    if val_loss < best_loss:
        best_loss = val_loss
        print('Saving Model...........')
        torch.save(model.state_dict(), 'model.pt')
    
    print(f'Epoch({epoch + 1}/{epochs}) -> Training Loss: {cum_loss/batches: .8f} | Validation Loss: {val_loss: .8f}')

    
                                                     
