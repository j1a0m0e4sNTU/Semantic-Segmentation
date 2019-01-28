import torch
import torch.nn as nn
import cv2 as cv
import numpy as np

if __name__ == '__main__':
    print('Testing ...')
    t = torch.zeros(4, 3, 100, 100)
    layer = nn.ConvTranspose2d(3, 3, kernel_size= 16, stride= 4, padding= 6, output_padding= 0)
    out = layer(t)
    print(out.size())