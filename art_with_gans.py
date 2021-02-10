import torch
import torchvision.transforms as tt
from torchvision.utils import make_grid
import torch.nn as nn
import matplotlib.pyplot as plt
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
stats=((0.5,0.5,0.5),(0.5,0.5,0.5))  
def to_device(data,device):
    if isinstance(data ,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device)


def denorm(img,mean,std):
    '''func to denormalize the image'''
    mean=torch.tensor(mean).reshape(1,3,1,1)
    std=torch.tensor(std).reshape(1,3,1,1)
    return img*std+mean
#generator model    
latent_size=128
print("Loading Model....")
generator=nn.Sequential(

    #in 1x1xlatent_size
    nn.ConvTranspose2d(latent_size,512,kernel_size=4,stride=1,padding=0,bias=False),
    #out 4x4xlatent_size
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    nn.ConvTranspose2d(512,256,kernel_size=4,stride=2,padding=1,bias=False),
    #out 8x8x256
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1,bias=False),
    # 16x16x128
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1,bias=False),
    #out 32x32x64
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    nn.ConvTranspose2d(64,3,kernel_size=4,stride=2,padding=1,bias=False),
    nn.Tanh()
    #out 64x64x3
    
)

model=to_device(generator,device)   #loading our trained model for inference
model.load_state_dict(torch.load('models/artgenerator_170epoch.pth'))

print("Generating Art....")
latent_vector=torch.randn(1,latent_size,1,1,device=device)
generated=generator(latent_vector)
img=torch.squeeze(denorm(generated.cpu().detach(),*stats))
img2 = tt.ToPILImage(mode='RGB')(img)

with open('./generated_art.jpg','wb') as art:
	img2.save(art)

print("Art Saved on a 64x64 canvas !!")
