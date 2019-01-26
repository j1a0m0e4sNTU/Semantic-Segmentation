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
    label =  (mask[0] > 128)*1
    label += (mask[1] > 128)*2
    label += (mask[2] > 128)*4
    return label.long()

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

def read_sat_image(path):
    '''
    read sat image and convert 
    '''
    img = cv.imread(path)
    img = numpy_to_torch(img, torch.float)
    return img

def read_mask_to_label(path):
    '''
    read mask and convert to label
    '''
    msk = cv.imread(path)
    msk = numpy_to_torch(msk, torch.long)
    label = mask_to_label(msk)
    return label

def process(img, name):
    '''
    convert mask.png to label map then convert back,
    save with name
    '''
    label = read_mask_to_label(img)
    mask = label_to_mask(label)
    mask = torch_to_numpy(mask)
    cv.imwrite(name, mask)

def mean_iou(label_x, label_y):
    '''
    type: torch
    labels: 1, 2, 3, 5, 6, 7
    label_x : prediction
    label_y : ground truth
    '''
    labels = [1, 2, 3, 5, 6, 7]
    iou = 0
    class_num = 0
    for label in labels:
        x = torch.sum(label_x == label).item()
        y = torch.sum(label_y == label).item()
        both = torch.sum((label_x == label) * (label_y == label)).item()
        if y > 0: # Calculate mean iou only when ground truth has that class
            iou += both/(x + y - both)
            class_num = class_num + 1

    mean_iou = iou/class_num
    return mean_iou

def mean_iou_batch(batch_x, batch_y):
    num = batch_x.size(0)
    iou = 0
    for i in range(num):
        iou = iou + mean_iou(batch_x[i], batch_y[i])
    iou = iou / num
    return iou

def test():
    label_1 = read_mask_to_label('0028_mask.png')
    label_2 = read_mask_to_label('0028_mask.png')
    m_iou = mean_iou(label_1, label_2)
    print('Mean IOU:', m_iou)

if __name__ == '__main__':
    process('0028_mask.png', 'test.png')