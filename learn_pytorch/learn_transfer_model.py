import torch
import torch.nn as nn
import numpy as np
import cv2

class convReLu(nn.Module):
    def __init__(self, inChannel, outChannel):
        super(convReLu, self).__init__()
        self.conv = nn.Conv2d(inChannel, outChannel, 3, 1, (1, 1))
    def forward(self, x):
        x = nn.functional.relu(self.conv(x))
        #print(x)
        return x

class Res50(nn.Module):
    def __init__(self):
        super(Res50, self).__init__()
        self.conv1 = convReLu(3, 5)
        self.conv2 = convReLu(5, 5)
        #self.out   = nn.Linear(5, 2, False)
        self.all = nn.Sequential(self.conv1, self.conv2)
        #self.list = nn.ModuleList()
        #self.all.add_module("conv1", module = self.conv1)
        #self.all.add_module(self.conv2)
        self.add_module("fc1", nn.Conv2d(5, 5, 5, 1, (0, 0)))
        self.add_module("out", nn.Linear(5, 2, False))

    def forward(self, x):        
        #x = self.conv1(x)
        #x = self.conv2(x)
        #x = self.out(x)
        x = self.all(x)
        x = self.fc1(x)
        #print(x.shape)
        return x


if __name__ == "__main__":
    a = torch.ones(1, 3, 5, 5)
    a[:, 1] = a[:, 1] * 2
    a[:, 2] = a[:, 2] * 3
    #print("before conv, \n", a)
    model = Res50()
    for name,para in model.named_parameters():
        print(name, '\n')
    for para in model.parameters():
        print(type(para), para.size())
    #print(model)
    output = model(a)
    #print("after conv, \n", output)
    #print(model.fc1.weight)
    
