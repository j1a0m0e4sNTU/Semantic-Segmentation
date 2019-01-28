import torch
import torch.nn as nn

def Conv2dLayer(in_planes, out_planes, kernrl_size = 3):
    layer = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size= kernrl_size, padding= (kernrl_size-1)//2),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace= True)
    )
    return layer

def ConvTranspose2dLayer(in_planes, out_planes, kernel_size, stride, padding= 0, output_padding= 0):
    layer = nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size= kernel_size, stride= stride, padding= padding, output_padding= output_padding),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace= True)
    )
    return layer

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.block_1 = nn.Sequential(
            Conv2dLayer(3, 64),
            Conv2dLayer(64, 64),
            nn.MaxPool2d((2, 2), stride= (2, 2))
        )
        self.block_2 = nn.Sequential(
            Conv2dLayer(64, 128),
            Conv2dLayer(128, 128),
            nn.MaxPool2d((2, 2), stride= (2, 2))
        )
        self.block_3 = nn.Sequential(
            Conv2dLayer(128, 256),
            Conv2dLayer(256, 256),
            Conv2dLayer(256, 256),
            nn.MaxPool2d((2, 2), stride= (2, 2))
        )
        self.block_4 = nn.Sequential(
            Conv2dLayer(256, 512),
            Conv2dLayer(512, 512),
            Conv2dLayer(512, 512),
            nn.MaxPool2d((2, 2), stride= (2, 2))
        )
        self.block_5 = nn.Sequential(
            Conv2dLayer(512, 512),
            Conv2dLayer(512, 512),
            Conv2dLayer(512, 512),
            nn.MaxPool2d((2, 2), stride= (2, 2))
        )

        self.upsample = ConvTranspose2dLayer(512, 64, kernel_size= 32, stride= 32, padding= 0,output_padding= 0)
        self.final_layer = nn.Conv2d(64, 8, kernel_size=1)

    def name(self):
        return 'FCN-16s (upsample with 1 step)'

    def forward(self, x): 
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.upsample(x)
        x = self.final_layer(x) 
        return x
        
def parameter_number(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def unit_test():
    batch_size = 8
    img_batch = torch.zeros(batch_size, 3, 512, 512)
    model = Model()
    out = model(img_batch)
    print('Input size: ', img_batch.size())
    print('Output size:', out.size())

def show_info():
    model = Model()
    print('Parameter number: ',parameter_number(model))
    print('Parameter structure:')
    print(model)
    
if __name__ == '__main__':
    #unit_test()
    #print('-- Pass unit test --')
    show_info()