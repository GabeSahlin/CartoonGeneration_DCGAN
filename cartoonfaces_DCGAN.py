import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import multiprocessing
from tqdm import tqdm

if __name__ == '__main__':
    multiprocessing.freeze_support()

    manualSeed = 999                        #random starting point for reproducibility : 999 initially chosen, but actual number doesn't matter
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True) #needed for reproducible results

    dataroot = r"C:\Cartoon50k"

    workers = 12                #number of worker threads
    batch_size = 64
    image_size = 256
    nc = 3                      #number of color channels in the input images. For color images this is 3
    nz = 100                    #length of latent vector
    ngf = 64                    #'Number of Generator Features' : number of channels in the generator's first transposed convolutional layer
    ndf = 64                    #'Number of Discriminator 'Features' :  number of channels in the discriminator's first convolutional layer
    num_epochs = 35
    lr = 0.0002                 #learning rate : currently 0.0002 for the Generator and 0.00005 for Discriminator
    beta1 = 0.5                 #hyperparameter for Adam optimizer : 0.5 is commonly used
    ngpu = 1

    #create the dataset
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    #create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    #determine if device can support cuda and gpu usage
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    #show some training images that was initially used
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()

    #all weights need to be randomly initialized with mean 0 and standard deviation 0.02 via DCGAN paper
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                #input latent vector nz, or z, of shape (nz) x 1 x 1 : nz = 1
                nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),  # (ngf*16) x 4 x 4 : (ngf*16) = 1024
                nn.BatchNorm2d(ngf * 16),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),  #(ngf*8) x 8 x 8 : (ngf*16) = 512
                nn.BatchNorm2d(ngf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),  #(ngf*4) x 16 x 16 : (ngf*16) = 256
                nn.BatchNorm2d(ngf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),  #(ngf*2) x 32 x 32 : (ngf*16) = 128
                nn.BatchNorm2d(ngf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),  #(ngf) x 64 x 64 : (ngf) = 64
                nn.BatchNorm2d(ngf),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),  #(ngf/2) x 128 x 128 : (ngf/2) = 32
                nn.BatchNorm2d(ngf // 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose2d(ngf // 2, nc, 4, 2, 1, bias=False),  #nc x 256 x 256 : nc = 3
                nn.Tanh()  #normalize output to [-1,1]
            )

        def forward(self, input):
            return self.main(input)

    #create the generator
    netG = Generator(ngpu).to(device)

    #use multiple gpus if available
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    #apply random weight initialization
    netG.apply(weights_init)

    #print built model
    #print(netG)


    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input nc x 256 x 256 : nc = 3
                nn.Conv2d(nc, ndf // 2, 4, 2, 1, bias=False),  #(ndf/2) x 128 x 128 : (ndf/2) = 32
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf // 2, ndf, 4, 2, 1, bias=False),  #(ndf) x 64 x 64 : (ndf) = 64
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), #(ndf*2) x 32 x 32 : (ndf*2) = 128
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),  #(ndf*4) x 16 x 16 : (ngf*4) = 256
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),  #(ndf*8) x 8 x 8 : (ndf*8) = 512
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),  # (ndf*16) x 4 x 4 : (ngf*16) = 1024
                nn.BatchNorm2d(ndf * 16),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),  #output : 1 x 1 x 1
                nn.Sigmoid() #probability [0,1] : fake/real
            )

        def forward(self, input):
            return self.main(input)

    #create discriminator
    netD = Discriminator(ngpu).to(device)

    # use multiple gpus if available
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # apply random weight initialization
    netD.apply(weights_init)

    # print built model
    #print(netD)

    #initialize loss function : Binary Cross Entropy
    criterion = nn.BCELoss()

    #create batch of latent vectors
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    #assign what a real label and what a fake label is
    real_label = .9
    fake_label = 0.

    #initialize Adam optimizers for generator and discriminator
    optimizerD = optim.Adam(netD.parameters(), lr=lr*.20, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    #lists to keep track of how the DCGAN is doing
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    #training loop
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        #progress bar to watch each epoch train
        pbar = tqdm(enumerate(dataloader, 0), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
        #for each batch
        for i, data in pbar:
            #train discriminator (first)

            #first with the real batch
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            #forward
            output = netD(real_cpu).view(-1)
            #find the loss from the real batch
            errD_real = criterion(output, label)
            #find the gradients used
            errD_real.backward()
            D_x = output.mean().item()

            #train with fake batch
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            #classify fake batch
            output = netD(fake.detach()).view(-1)
            #find the loss from the fake batch
            errD_fake = criterion(output, label)
            #find the gradients used, added with the previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            #find total loss/error/penalty for discriminator over the fake and real batches
            errD = errD_real + errD_fake
            #update discriminator
            optimizerD.step()

            #train generator (second)
            netG.zero_grad()
            label.fill_(real_label)
            #forward
            output = netD(fake).view(-1)
            #find loss of generator based on output
            errG = criterion(output, label)
            #find gradients used
            errG.backward()
            D_G_z2 = output.mean().item()
            #udpate generator
            optimizerG.step()

            #print losses
            pbar.set_postfix({
                'Loss_D': f'{errD.item():.4f}',
                'Loss_G': f'{errG.item():.4f}',
                'D(x)': f'{D_x:.4f}',
                'D(G(z))': f'{D_G_z1:.4f} / {D_G_z2:.4f}'
            })

            #save the values to the lists
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            iters += 1

        #output each epoch of generated images
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        img = vutils.make_grid(fake, padding=2, normalize=True)
        img_list.append(img)

        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title(f"Generated Images After Epoch {epoch + 1}")
        plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.show()

        print(f"Epoch {epoch + 1} complete.\n")

    # plot the losses of each network over all iterations
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    #animation the epochs together
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    HTML(ani.to_jshtml())

    #initial image next to final image
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()
