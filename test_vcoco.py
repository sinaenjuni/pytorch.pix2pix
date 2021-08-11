
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from dataset.vcoco import VCOCODataset, PairedDataRandomHorizontalFlip
from models import UnetGenerator
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from pathlib import Path


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def toCPU(x):
    return x.data.cpu()

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
                          batch_size=1,
                          shuffle=False,
                          num_workers=int(1)
                          )

device = torch.device('cuda:{}'.format(0))



train_iter = iter(train_loader)




netG_64 = UnetGenerator(input_nc=3,
                     output_nc=3,
                     num_downs=8,
                     ngf=64,
                     norm_layer=nn.BatchNorm2d,
                     use_dropout=True).to(device)

PATH = './save_model4(vcoco)/epoch_200.pt'
netG_64.load_state_dict(torch.load(PATH))
netG_64.eval()


netG_128 = UnetGenerator(input_nc=3,
                     output_nc=3,
                     num_downs=8,
                     ngf=128,
                     norm_layer=nn.BatchNorm2d,
                     use_dropout=True).to(device)

PATH = './save_model3(vcoco)/epoch_200.pt'
netG_128.load_state_dict(torch.load(PATH))
netG_128.eval()



for i in range(len(train_loader.dataset)):
    data = train_iter.next()
    real_A = data['A'].to(device)
    real_B = data['B'].to(device)

    # plt.imshow(real_A[0].permute(1,2,0).data.cpu())
    # plt.show()
    # plt.imshow(real_B[0].permute(1,2,0).data.cpu())
    # plt.show()

    gened_A_64 = netG_64(real_B)
    gened_A_128 = netG_128(real_B)

    grid = torch.cat([toCPU(real_B), toCPU(real_A), toCPU(gened_A_64), toCPU(gened_A_128)], 3)
    grid = make_grid(denorm(grid), nrow=1)
    save_path = Path(f'./test_sample/64+128/{i}.jpg')
    if not save_path.parent.exists():
        save_path.parent.mkdir(exist_ok=True, parents=True)
    save_image(grid, save_path, nrow=1)

    # plt.imshow(grid.permute(1, 2, 0))
    # plt.show()