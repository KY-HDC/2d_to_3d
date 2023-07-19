import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.autograd import Variable

from dataset import Custom_Dataset
from network import Generator, Discriminator
import argparse

os.makedirs("result_images", exist_ok=True)

def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
        parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
        parser.add_argument("--lr", type=float, default=0.00005, help="adam: learning rate")
        parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
        parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
        parser.add_argument("--data", type=str, default='/data2/gayrat/vs-projects/MY_GANs/datasets/CT_64', help="path to data")
        parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
        parser.add_argument("--channels", type=int, default=1, help="number of image channels")
        parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
        parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
        parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
        opt = parser.parse_args()

        cuda = True if torch.cuda.is_available() else False

        # CT dataset
        dataset_train = Custom_Dataset(data_dir=opt.data)

        # Data loader
        data_loader = DataLoader(dataset=dataset_train,
                                batch_size=opt.batch_size, 
                                shuffle=True)


        generator = Generator(opt.latent_dim, opt.channels, opt.img_size)
        discriminator = Discriminator(opt.channels, opt.img_size)

        if cuda:
            generator.cuda()
            discriminator.cuda()

        # Optimizers
        optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
        optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        # ----------
        #  Training
        # ----------

        batches_done = 0
        for epoch in range(opt.n_epochs):

            for i, imgs in enumerate(data_loader):

                # Configure input
                real_imgs = Variable(imgs.type(Tensor))

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

                # Generate a batch of images
                fake_imgs = generator(z).detach()
                # Adversarial loss
                loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

                loss_D.backward()
                optimizer_D.step()

                # Clip weights of discriminator
                for p in discriminator.parameters():
                    p.data.clamp_(-opt.clip_value, opt.clip_value)

                # Train the generator every n_critic iterations
                if i % opt.n_critic == 0:

                    # -----------------
                    #  Train Generator
                    # -----------------

                    optimizer_G.zero_grad()

                    # Generate a batch of images
                    gen_imgs = generator(z)
                    # Adversarial loss
                    loss_G = -torch.mean(discriminator(gen_imgs))

                    loss_G.backward()
                    optimizer_G.step()

                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch, opt.n_epochs, batches_done % len(data_loader), len(data_loader), loss_D.item(), loss_G.item())
                    )

                if batches_done % opt.sample_interval == 0:
                    save_image(gen_imgs.data[:25], "result_images/%d.png" % batches_done, nrow=5, normalize=True)
                batches_done += 1

        # Save the model checkpoints 
        # torch.save(generator.state_dict(), 'G.ckpt')
        # torch.save(discriminator.state_dict(), 'D.ckpt')
if __name__ == '__main__':
	main()