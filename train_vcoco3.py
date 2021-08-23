# import os
# import sys
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# print()

import torch
import torch.nn as nn
import torchvision.transforms as T

from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision.utils import save_image, make_grid

import loss
from models import UnetGenerator
from models import NLayerDiscriminator

from dataset.vcoco import  VCOCODataset, PairedDataRandomHorizontalFlip

import time
from pathlib import Path

from torch.backends import cudnn
cudnn.benchmark = True

print(torch.cuda.is_available())

SAMPLEPATH = 'samples'

lr = 0.0001
beta1 = 0.5
# gan_mode = 'vanilla'
gan_mode = 'lsgan'
batch_size = 16
serial_batches = False
num_threads = 4

start_epoch = 201
end_epoch = 500

lr_decay_point = 100
lr_decay_rate = 100
lambda_L1 = 10.0
lambda_gp = 10.0

gpu_ids = 1
# device = torch.device('cuda:{}'.format(gpu_ids)) if gpu_ids else torch.device('cpu')
device = torch.device('cuda:{}'.format(gpu_ids))


def gradient_penalty(fake, real):

    alpha = torch.rand(real.size(0), 1, 1, 1).to(device)
    interpolated_sample = ((alpha * real.data) + ((alpha - 1) * fake.data)).requires_grad_(True)
    interpolated_pred = netD(interpolated_sample)


    weight = torch.ones(interpolated_pred.size()).to(device)
    dydx = torch.autograd.grad(outputs=interpolated_pred,
                               inputs=interpolated_sample,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
    return torch.mean((dydx_l2norm - 1) ** 2)



def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)



def save_images(path, list_tensor):
    save_path = Path(path)
    if not save_path.parent.exists():
        save_path.parent.mkdir(exist_ok=True, parents=True)
    cat_data = torch.cat(list_tensor, dim=3)
    cat_data = denorm(cat_data.data.cpu())
    save_image(cat_data, save_path, nrow=1)

def save_model(path):
    save_path = Path(path)
    if not save_path.parent.exists():
        save_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(netG.state_dict(), path.replace('.pt', 'G.pt'))
    torch.save(netD.state_dict(), path.replace('.pt', 'D.pt'))


def lambda_rule(epoch):
    lr_l = 1.0 - max(0, epoch - (end_epoch - lr_decay_point)) / float(lr_decay_rate + 1)
    return lr_l


def update_learning_rate():
    """Update learning rates for all the networks; called at the end of every epoch"""
    old_lr = optimizers[0].param_groups[0]['lr']
    for scheduler in schedulers:
          scheduler.step()
    lr = optimizers[0].param_groups[0]['lr']
    print('learning rate %.7f -> %.7f' % (old_lr, lr))

transforms = T.Compose([
                        T.ToTensor(),
                        T.Resize((256, 256)),
                        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        # T.Normalize((0.5), (0.5)),
                    ])
paired_transforms = T.Compose([
                               PairedDataRandomHorizontalFlip(0.5)
                               # PairedDataRandomCrop()
                            ])

train_dataset = VCOCODataset('./data/vcoco/images/train2014',
                             './data/vcoco/segmentation/train2014',
                             transforms, paired_transforms)
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=not serial_batches,
                          num_workers=int(num_threads)
                          )


# val_dataset = VCOCODataset('./data/vcoco/images/val2014',
#                            './data/vcoco/segmentation/val2014',
#                              transforms, paired_transforms)
# val_loader = DataLoader(val_dataset,
#                         batch_size=4,
#                         shuffle=False)
# val_iter = iter(val_loader)
# val_data = val_iter.next()

fix_iter = iter(train_loader)
fix_data = fix_iter.next()

fixed_Adata = fix_data['A'].to(device)
fixed_Bdata = fix_data['B'].to(device)


# fixed_data = []
# fixed_data.append(fix_data["B"].to(device))
# fixed_data.append(fix_data["A"].to(device))

# fix_data = []
# fix_data.append()
# fix_data = torch.cat([fix_data['B'].to(device), fix_data['A'].to(device)], 3)

# print(fix_data)



# print(A)
import matplotlib.pyplot as plt
# plt.imshow(fixed_data)
# plt.show()
# plt.imshow(make_grid(denorm(fix_data), nrow=1).permute(1, 2, 0))
# plt.show()
# plt.imshow(_data['B'].permute(1, 2, 0) + 1 / 2)
# plt.show()



netG = UnetGenerator(input_nc=3,
                     output_nc=3,
                     num_downs=8,
                     ngf=64,
                     norm_layer=nn.BatchNorm2d,
                     use_dropout=True).to(device)
print(netG)

netD = NLayerDiscriminator(input_nc=3 + 3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d).to(device)
print(netD)


if start_epoch != 1:
    netD.load_state_dict(torch.load(f'./save_files/model4(vcoco)/epoch_{start_epoch-1}D.pt'))
    netG.load_state_dict(torch.load(f'./save_files/model4(vcoco)/epoch_{start_epoch-1}G.pt'))


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

adversarial_loss = torch.nn.MSELoss().to(device)


for epoch in range(start_epoch, end_epoch + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()  # timer for data loading per iteration
    epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch

    update_learning_rate()  # update learning rates in the beginning of every epoch.
    ret_print['eopch'] = epoch
    for i, data in enumerate(train_loader):  # inner loop within one epoch
        ret_print['iter'] = i
        iter_start_time = time.time()  # timer for computation per iteration
        if total_iters % print_freq == 0:
            t_data = iter_start_time - iter_data_time

        total_iters += batch_size
        epoch_iter += batch_size

        AtoB = direction == 'AtoB'
        real_A = data['A' if AtoB else 'B'].to(device)
        real_B = data['B' if AtoB else 'A'].to(device)
        # image_paths = data['A_paths' if AtoB else 'B_paths']

        valid = torch.ones(real_A.size(0), 1)
        fake  = torch.zeros(real_A.size(0), 1)


        # for param in netD.parameters():
        #     param.requires_grad = True
        fake_B = netG(real_A)

        fake_AB = torch.cat((real_A, fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = netD(fake_AB.detach())
        loss_D_fake = criterionGAN(pred_fake, False)
        loss_D_fake = torch.mean(pred_fake)
        # loss_D_fake = adversarial_loss(pred_fake, valid)

        # Real
        real_AB = torch.cat((real_A, real_B), 1)
        pred_real = netD(real_AB)
        loss_D_real = criterionGAN(pred_real, True)
        # loss_D_real = -torch.mean(pred_real)
        # loss_D_real = adversarial_loss(pred_fake, fake)

        # loss_gp = gradient_penalty(fake_AB, real_AB)

        # combine loss and calculate gradients
        # loss_D = (loss_D_fake + loss_D_real) * 0.5
        # loss_D = loss_D_fake + loss_D_real + (lambda_gp * loss_gp)
        loss_D = 0.5 * (loss_D_fake + loss_D_real)


        optimizer_D.zero_grad()     # set D's gradients to zero
        loss_D.backward()
        optimizer_D.step()          # update D's weight

        ret_print['loss_D_fake'] = loss_D_fake.item() * real_A.shape[0]
        ret_print['loss_D_real'] = loss_D_real.item() * real_A.shape[0]
        # ret_print['loss_gp'] = lambda_gp * loss_gp.item()

        # for param in netD.parameters():
        #     param.requires_grad = False

        optimizer_G.zero_grad()        # set G's gradients to zero
        fake_B = netG(real_A)

        fake_AB = torch.cat((real_A, fake_B), 1)
        pred_fake = netD(fake_AB.detach())
        loss_G_GAN = criterionGAN(pred_fake, True)
        # loss_G_GAN = -torch.mean(pred_fake)
        # loss_G_GAN = adversarial_loss(pred_fake, valid)

        # Second, G(A) = B
        loss_G_L1 = criterionL1(fake_B, real_B) * lambda_L1
        # combine loss and calculate gradients
        loss_G = loss_G_GAN + loss_G_L1
        loss_G.backward()
        optimizer_G.step()             # udpate G's weights


        ret_print["loss_G_GAN"] = loss_G_GAN.item() * real_A.shape[0]
        ret_print["loss_G_L1"]  = loss_G_L1.item()  * real_A.shape[0]

        print(str(ret_print))

    with torch.no_grad():
        sav_img = [fixed_Bdata, fixed_Adata, netG(fix_data['B'].to(device))]
        save_images(f'./save_files/samples4(vcoco)/epoch_{epoch}.png', sav_img)
        save_model(f'./save_files/model4(vcoco)/epoch_{epoch}.pt')

