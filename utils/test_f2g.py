import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from PIL import Image
import click

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 1, 3, padding=1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.act(x)
        return x
@click.command()
@click.option('--model_path', default="f2g_trained.pt", help='Trained model path')  
@click.option('--input_path', default="../trainsets/earth1_samples/earth1_samples_trained/", help="Input npz path") 
def test(model_path, input_path):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu") 
    PATH = model_path
    input_path = input_path
    model = Net()
    model.to(device)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    
    with torch.no_grad():
        input_path = np.load(input_path)
        image = input_path.f.arr_0
        im1 = Image.fromarray(np.array(image), mode='F')
        im1.save("before_normalization.jpg")
        image = torch.tensor(image).reshape((1,1,256,256))
        image = (image/300890.0)*3
        im2 = Image.fromarray(image, mode='F')
        im2.save("after_normalization.jpg")
        image = image.to(device)
        output = model(image)
        print(output.shape)
        im3 = Image.fromarray(np.array(output), mode='F')
        im3.save("output.jpg")
if __name__ == "__main__":
    test()
        
        
