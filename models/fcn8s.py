import torch
import torch.nn as nn

def Conv2dLayer(in_planes, out_planes, kernrl_size = 3):
    layer = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size= kernrl_size, padding= (kernrl_size-1)//2),
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

        self.relu = nn.ReLU(inplace= True)
        self.deconv_1 = nn.ConvTranspose2d(512, 512, kernel_size= 3, stride= 2, padding= 1, output_padding= 1)
        self.bn_1   = nn.BatchNorm2d(512)
        self.deconv_2 = nn.ConvTranspose2d(512, 256, kernel_size= 3, stride= 2, padding= 1, output_padding= 1)
        self.bn_2   = nn.BatchNorm2d(256)
        self.deconv_3 = nn.ConvTranspose2d(256, 128, kernel_size= 3, stride= 2, padding= 1, output_padding= 1)
        self.bn_3   = nn.BatchNorm2d(128)
        self.deconv_4 = nn.ConvTranspose2d(128, 64, kernel_size= 3, stride= 2, padding= 1, output_padding= 1)
        self.bn_4   = nn.BatchNorm2d(64)
        self.deconv_5 = nn.ConvTranspose2d(64, 32, kernel_size= 3, stride= 2, padding= 1, output_padding= 1)
        self.bn_5   = nn.BatchNorm2d(32)

        self.final_layer = nn.Conv2d(32, 8, kernel_size=1)

    def name(self):
        return 'FCN-8s'

    def forward(self, x): 
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        s_8 = x 
        x = self.block_4(x)
        s_16 = x
        x = self.block_5(x)
        
        x = self.relu(self.bn_1(self.deconv_1(x)))
        x = x + s_16
        x = self.relu(self.bn_2(self.deconv_2(x)))
        x = x + s_8
        x = self.relu(self.bn_3(self.deconv_3(x)))
        x = self.relu(self.bn_4(self.deconv_4(x)))
        x = self.relu(self.bn_5(self.deconv_5()))
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
    unit_test()
    print('-- Pass unit test --')
    #show_info()