import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(3,  16, kernel_size= 3, padding= 1),
            nn.ReLU(inplace= True),
            nn.Conv2d(16, 32, kernel_size= 3, padding= 1),
            nn.ReLU(inplace= True),
            nn.MaxPool2d((4, 4), stride=(4, 4))
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size= 3, padding= 1),
            nn.ReLU(inplace= True),
            nn.Conv2d(64, 128, kernel_size= 3, padding= 1),
            nn.ReLU(inplace= True),
            nn.MaxPool2d((4, 4), stride=(4, 4))
        )

        self.block_3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size= 2, stride= 2),
            nn.ReLU(inplace= True),
            nn.ConvTranspose2d(64,  64, kernel_size= 2, stride= 2),
            nn.ReLU(inplace= True),
            nn.ConvTranspose2d(64,  32, kernel_size= 2, stride= 2),
            nn.ReLU(inplace= True),
            nn.ConvTranspose2d(32,   8, kernel_size= 2, stride= 2)
        )
    
    def name(self):
        return 'base_01'

    def forward(self, x): 
        x = self.block_1(x) 
        x = self.block_2(x) 
        x = self.block_3(x)
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