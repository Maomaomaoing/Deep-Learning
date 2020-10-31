import itertools
import os

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

import time
start_time = time.time()

if not os.path.exists('ckpt'):
    os.makedirs('ckpt')

def save_json(data, filename):
    import json
    to_float = lambda num: float(num)
    with open(filename, 'w') as f:
        json.dump(list(map(to_float, data)), f)
    print("Save file at", filename)

# parameters
#TODO : set up all the parameters
epochs =  100   # number of epochs of training
batchsize =  50   # size of the batches
animation_root = os.getcwd()+'/dataset_animation'    # root directory of the dataset
cartoon_root = os.getcwd()+'/dataset_cartoon'    # root directory of the dataset
lr1 = 1e-4    # initial learning rate
lr2 = 1e-5
size = 64    # size of the data crop (squared assumed)
input_nc = 3    # number of channels of input data
output_nc = 3    # number of channels of output data

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###### Definition of variables ######
# Networks
netG_A2B = Generator(input_nc, output_nc)
netG_B2A = Generator(output_nc, input_nc)
netD_A = Discriminator(input_nc)
netD_B = Discriminator(output_nc)

netG_A2B.to(device)
netG_B2A.to(device)
netD_A.to(device)
netD_B.to(device)

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()


# Optimizers
#optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=lr1, betas=(0.5, 0.999))
optimizer_G = torch.optim.RMSprop(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=lr1)
#optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=lr2, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.RMSprop(netD_A.parameters(), lr=lr2)
#optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=lr2, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.RMSprop(netD_B.parameters(), lr=lr2)


# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
input_A = Tensor(batchsize, input_nc, size, size)
input_B = Tensor(batchsize, output_nc, size, size)
target_real = Variable(Tensor(batchsize,1).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(batchsize,1).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
transform1 = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform2 = transforms.Compose([transforms.CenterCrop((300, 300)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

animation_set = torchvision.datasets.ImageFolder(animation_root, transform1) 
cartoon_set = torchvision.datasets.ImageFolder(cartoon_root, transform2)

animation_loader = torch.utils.data.DataLoader(dataset=animation_set,batch_size=batchsize,shuffle=True)
cartoon_loader = torch.utils.data.DataLoader(dataset=cartoon_set,batch_size=batchsize,shuffle=True)
###################################
G_loss  = []
DA_loss  = []
DB_loss  = []
###### Training ######
for epoch in range(1, epochs):
    i=1
    print('epoch',epoch)
    for j, batch in enumerate(zip(animation_loader, cartoon_loader)):
        # Set model input
        A = torch.FloatTensor(batch[0][0])
        B = torch.FloatTensor(batch[1][0])
        real_A = Variable(input_A.copy_(A))
        real_B = Variable(input_B.copy_(B))

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()
        
        copy_B = netG_A2B(real_B)
        copy_A = netG_B2A(real_A)
        
        generated_B = netG_A2B(real_A)
        generated_A = netG_B2A(real_B)
        
        backto_A = netG_B2A(generated_B)
        backto_B = netG_A2B(generated_A)
        
        predict_A = netD_A(generated_A)
        predict_B = netD_B(generated_B)
        
        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        # TODO : calculate the loss for the generators, and assign to loss_G
        loss_G = criterion_identity(copy_A, real_A) + criterion_identity(copy_B, real_B) + criterion_GAN(predict_A, target_real) + criterion_GAN(predict_B, target_real) + criterion_cycle(backto_A, real_A) + criterion_cycle(backto_B, real_B)
        loss_G.backward()
        
        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()
        
        predict_real = netD_A(real_A)
        predict_fake = netD_A( fake_A_buffer.push_and_pop(generated_A).detach() )

        # TODO : calculate the loss for a discriminator, and assign to loss_D_A
        loss_D_A = criterion_GAN(predict_real, target_real) + criterion_GAN(predict_fake, target_fake)
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()
        
        predict_real = netD_B(real_B)
        predict_fake = netD_B(fake_B_buffer.push_and_pop(generated_B).detach())

        # TODO : calculate the loss for the other discriminator, and assign to loss_D_B
        loss_D_B = criterion_GAN(predict_real, target_real) + criterion_GAN(predict_fake, target_fake)
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################
        
        # Progress report
        if (i%100==0):
            print("loss_G : ",loss_G.data.cpu().numpy() ,",loss_D:", (loss_D_A.data.cpu().numpy() + loss_D_B.data.cpu().numpy()))
            i=0
        G_loss.append(loss_G.item())
        DA_loss.append(loss_D_A.item())
        DB_loss.append(loss_D_B.item())
        i=i+1
    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), 'ckpt/netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), 'ckpt/netG_B2A.pth')
    torch.save(netD_A.state_dict(), 'ckpt/netD_A.pth')
    torch.save(netD_B.state_dict(), 'ckpt/netD_B.pth')
    
save_json(G_loss, "G_loss.json")
save_json(DA_loss, "DA_loss.json")
save_json(DB_loss, "DB_loss.json")
    
end_time = time.time()
print('Total cost time',time.strftime("%H hr %M min %S sec", time.gmtime(end_time - start_time)))

# TODO : plot the figure
