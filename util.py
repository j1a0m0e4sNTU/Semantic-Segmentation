import numpy as np
import cv2 as cv
import torch

def numpy_to_torch(array, dtype):
    '''
    convert array to tensor, and change chanel order.
    '''
    t = torch.tensor(array, dtype= dtype)
    t = t.permute(2, 0, 1)
    return t

def torch_to_numpy(tensor):
    '''
    convert tensor to array, and change chanel order.
    '''
    tensor = tensor.permute(1, 2, 0)
    a = tensor.numpy() 
    return a

def mask_to_label(mask):
    '''
    type: troch.tensor
    convert mask to label map (1 chanel)
    '''
    label =  (mask[0]//255)*1
    label += (mask[1]//255)*2
    label += (mask[2]//255)*4
    return label

def label_to_mask(label):
    '''
    type: torch.tensor
    convert label map to mask (3 chanels)
    '''
    w, h = label.size()
    mask = torch.zeros((3, w, h), dtype= torch.long)
    mask[0] = label%2
    label = label//2
    mask[1] = label%2
    label = label//2
    mask[2] = label%2

    mask = mask * 255
    return mask

def process(img, name):
    '''
    convert mask.png to label map then convert back,
    save with name
    '''
    img = cv.imread(img)
    img = numpy_to_torch(img, dtype=torch.long)
    label = mask_to_label(img)
    mask = label_to_mask(label)
    mask = torch_to_numpy(mask)
    cv.imwrite(name, mask)

def test():
    pass

if __name__ == '__main__':
    test()