import numpy as np
import torch
from torch import nn
from torch import optim
from torch import autograd
from torch.autograd import Variable
import nibabel as nib
from torch.utils.data import DataLoader
from nilearn import plotting
from dataset import Custom_Dataset
from network import Generator, Discriminator


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = '/data2/gayrat/vs-projects/MY_GANs/datasets/CT-0'

BATCH_SIZE=4
max_epoch = 100
lr = 0.0001
gpu = True
workers = 4

LAMBDA= 10

latent_dim = 1000

trainset = Custom_Dataset(data_dir=data_dir)
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,num_workers=workers)


def inf_train_gen(data_loader):
    while True:
        for _,images in enumerate(data_loader):
            yield images


D = Discriminator().to(device)
G = Generator().to(device)

D = nn.DataParallel(D)
G = nn.DataParallel(G)

g_optimizer = optim.Adam(G.parameters(), lr=0.0002)
d_optimizer = optim.Adam(D.parameters(), lr=0.0002)


def calc_gradient_penalty(netD, real_data, fake_data):    
    alpha = torch.rand(real_data.size(0),1,1,1,1)
    alpha = alpha.expand(real_data.size())
    
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


real_y = Variable(torch.ones((BATCH_SIZE, 1)).cuda())
fake_y = Variable(torch.zeros((BATCH_SIZE, 1)).cuda())
loss_f = nn.BCELoss()

d_real_losses = list()
d_fake_losses = list()
d_losses = list()
g_losses = list()
divergences = list()

TOTAL_ITER = 200000
gen_load = inf_train_gen(train_loader)

count=0
for iteration in range(TOTAL_ITER):
    ###############################################
    # Train D 
    ###############################################
    for p in D.parameters():  
        p.requires_grad = True 

    real_images = gen_load.__next__()
    D.zero_grad()
    real_images = Variable(real_images).cuda()
    # print(real_images.shape) # torch.Size([4, 128, 128, 64])

    _batch_size = real_images.size(0)

    y_real_pred = D(real_images)

    d_real_loss = y_real_pred.mean()
    
    noise = Variable(torch.randn(_batch_size, latent_dim, 1, 1, 1)).cuda()
    # print(noise.shape) # torch.Size([4, 1000, 1, 1, 1])

    fake_images = G(noise)
    y_fake_pred = D(fake_images.detach())

    d_fake_loss = y_fake_pred.mean()

    gradient_penalty = calc_gradient_penalty(D,real_images.data, fake_images.data)
 
    d_loss = - d_real_loss + d_fake_loss +gradient_penalty
    d_loss.backward()
    Wasserstein_D = d_real_loss - d_fake_loss

    d_optimizer.step()

    ###############################################
    # Train G 
    ###############################################
    for p in D.parameters():
        p.requires_grad = False
        
    for iters in range(5):
        G.zero_grad()
        noise = Variable(torch.randn((_batch_size, latent_dim, 1, 1 ,1)).cuda())
        fake_image =G(noise)
        y_fake_g = D(fake_image)

        g_loss = -y_fake_g.mean()

        g_loss.backward()
        g_optimizer.step()

    ###############################################
    # Visualization
    ###############################################
    
    if iteration%10 == 0:
        d_real_losses.append(d_real_loss.data)
        d_fake_losses.append(d_fake_loss.data)
        d_losses.append(d_loss.data)
        g_losses.append(g_loss.data.cpu().numpy())

        print('[{}/{}]'.format(iteration,TOTAL_ITER),
              'D: {:<8.3}'.format(d_loss.data.cpu().numpy()), 
              'D_real: {:<8.3}'.format(d_real_loss.data.cpu().numpy()),
              'D_fake: {:<8.3}'.format(d_fake_loss.data.cpu().numpy()), 
              'G: {:<8.3}'.format(g_loss.data.cpu().numpy()))

        featmask = np.squeeze((0.5*fake_image+0.5)[0].data.cpu().numpy())

        featmask = nib.Nifti1Image(featmask,affine = np.eye(4))
        plotting.plot_img(featmask,title="FAKE")
        # plotting.show()
        
    if (iteration+1)%500 ==0:
        torch.save(G.state_dict(),'./checkpoint/G_W_iter'+str(iteration+1)+'.pth')
        torch.save(D.state_dict(),'./checkpoint/D_W_iter'+str(iteration+1)+'.pth')