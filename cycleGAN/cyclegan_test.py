from torchvision.utils import save_image
import os
import sys

import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import torchvision
import numpy as np

from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import weights_init_normal
###### Definition of variables ######
# TODO : assign input_nc and output_nc
input_nc = 3
output_nc = 3
size = 64    # size of the data crop (squared assumed)
batchsize =  50   # size of the batches
animation_root = os.getcwd()+'/dataset_animation'    # root directory of the dataset
cartoon_root = os.getcwd()+'/dataset_cartoon'    # root directory of the dataset


# Networks
netG_A2B = Generator(input_nc, output_nc)
netG_B2A = Generator(output_nc, input_nc)

# Load state dicts
netG_A2B.load_state_dict(torch.load('ckpt/netG_A2B.pth'))
netG_B2A.load_state_dict(torch.load('ckpt/netG_B2A.pth'))

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
netG_A2B.to(device)
netG_B2A.to(device)

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

# Dataset loader
transform1 = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform2 = transforms.Compose([transforms.CenterCrop((300, 300)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
animation_set = torchvision.datasets.ImageFolder(animation_root, transform1) 
cartoon_set = torchvision.datasets.ImageFolder(cartoon_root, transform2) 
animation_loader = torch.utils.data.DataLoader(dataset=animation_set,batch_size=batchsize,shuffle=True)
cartoon_loader = torch.utils.data.DataLoader(dataset=cartoon_set,batch_size=batchsize,shuffle=True)


if not os.path.exists('output/animation'):
    os.makedirs('output/animation')
if not os.path.exists('output/cartoon'):
    os.makedirs('output/cartoon')

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
input_A = Tensor(batchsize, input_nc, size, size)
input_B = Tensor(batchsize, output_nc, size, size)
i=0
for batch in zip(animation_loader, cartoon_loader):
    # Set model input
    A = torch.FloatTensor(batch[0][0])
    B = torch.FloatTensor(batch[1][0])
    real_A = Variable(input_A.copy_(A))
    real_B = Variable(input_B.copy_(B))

    # Generate output
    fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
    fake_A = 0.5*(netG_B2A(real_B).data + 1.0)

    # Save image files
    save_image(real_A, 'output/animation/real%04d.png' % (i+1))
    save_image(real_B, 'output/cartoon/real%04d.png' % (i+1))
    save_image(fake_A, 'output/animation/fake%04d.png' % (i+1))
    save_image(fake_B, 'output/cartoon/fake%04d.png' % (i+1))

    sys.stdout.write('\rGenerated images %04d' % (i+1))
    i = i+1
    if (i==10):
        break

sys.stdout.write('\n')
