import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch.optim import lr_scheduler

import loss
from models import UnetGenerator
from models import NLayerDiscriminator

from dataset import FacadeDataset
import time

SAMPLEPATH = 'samples'

lr = 0.0002
beta1 = 0.5
gan_mode = 'vanilla'
batch_size = 1
serial_batches = False
num_threads = 4
epoch_count = 1
n_epochs = 100
n_epochs_decay = 100
lambda_L1 = 100.0

gpu_ids = 0
device = torch.device('cuda:{}'.format(gpu_ids)) if gpu_ids else torch.device('cpu')

def lambda_rule(epoch):
    lr_l = 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1)
    return lr_l


def update_learning_rate():
    """Update learning rates for all the networks; called at the end of every epoch"""
    old_lr = optimizers[0].param_groups[0]['lr']
    for scheduler in schedulers:
          scheduler.step()
    lr = optimizers[0].param_groups[0]['lr']
    print('learning rate %.7f -> %.7f' % (old_lr, lr))



train_dataset = FacadeDataset('./data/facade/train')
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=not serial_batches,
                          num_workers=int(num_threads)
                          )
_data = train_loader.dataset[10]

# print(A)
# import matplotlib.pyplot as plt
#
# plt.imshow(_data['A'].permute(1, 2, 0) + 1 / 2)
# plt.show()
# plt.imshow(_data['B'].permute(1, 2, 0) + 1 / 2)
# plt.show()



netG = UnetGenerator(input_nc=3,
                     output_nc=3,
                     num_downs=8,
                     ngf=64,
                     norm_layer=nn.BatchNorm2d,
                     use_dropout=True)
print(netG)

netD = NLayerDiscriminator(input_nc=3 + 3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d)
print(netD)



criterionGAN = loss.GANLoss(gan_mode).to(device)
criterionL1 = torch.nn.L1Loss()
# initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
optimizer_G = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizers = []
optimizers.append(optimizer_G)
optimizers.append(optimizer_D)

schedulers = [lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule) for optimizer in optimizers]


total_iters = 0
print_freq = 100
direction = 'BtoA'
ret_print = {}

for epoch in range(epoch_count, n_epochs + n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()  # timer for data loading per iteration
    epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

    update_learning_rate()  # update learning rates in the beginning of every epoch.

    for i, data in enumerate(train_loader):  # inner loop within one epoch
        iter_start_time = time.time()  # timer for computation per iteration
        if total_iters % print_freq == 0:
            t_data = iter_start_time - iter_data_time

        total_iters += batch_size
        epoch_iter += batch_size

        AtoB = direction == 'AtoB'
        real_A = data['A' if AtoB else 'B'].to(device)
        real_B = data['B' if AtoB else 'A'].to(device)
        image_paths = data['A_paths' if AtoB else 'B_paths']

        fake_B = netG(real_A)

        optimizer_D.zero_grad()     # set D's gradients to zero
        fake_AB = torch.cat((real_A, fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = netD(fake_AB.detach())
        loss_D_fake = criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((real_A, real_B), 1)
        pred_real = netD(real_AB)
        loss_D_real = criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_D.backward()
        optimizer_D.step()          # update D's weight

        ret_print['loss_D_fake'] = loss_D_fake.item() * real_A.shape[0]
        ret_print['loss_D_real'] = loss_D_real.item() * real_A.shape[0]

        optimizer_G.zero_grad()        # set G's gradients to zero
        fake_AB = torch.cat((real_A, fake_B), 1)
        pred_fake = netD(fake_AB)
        loss_G_GAN = criterionGAN(pred_fake, True)
        # Second, G(A) = B
        loss_G_L1 = criterionL1(fake_B, real_B) * lambda_L1
        # combine loss and calculate gradients
        loss_G = loss_G_GAN + loss_G_L1
        loss_G.backward()
        optimizer_G.step()             # udpate G's weights

        ret_print["loss_G_GAN"] = loss_G_GAN.item() * real_A.shape[0]
        ret_print["loss_G_L1"] = loss_G_L1.item() * real_A.shape[0]

        print(str(ret_print))


