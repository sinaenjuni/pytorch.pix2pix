import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid

import random
from pathlib import Path
from PIL import Image

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


class PairedDataRandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, imgs):
        img1 = imgs[0]
        img2 = imgs[1]
        if random.random() > self.prob:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)
        # return (img1, img2)
        return img1, img2

class PairedDataRandomCrop:
    def __init__(self, prob=0.5, output_size=(256, 256)):
        self.prob = prob
        self.output_size = output_size

    def __call__(self, imgs):
        img1 = imgs[0]
        img2 = imgs[1]
        if random.random() > self.prob:
            i, j, h, w = T.RandomCrop.get_params(img1, output_size=self.output_size)
            img1 = TF.crop(img1, i, j, h, w)
            img2 = TF.crop(img2, i, j, h, w)
        return (img1, img2)


class VCOCODataset(Dataset):
    def __init__(self, img_path, seg_path, transforms=None, paired_transforms=None):
        self.img_path = Path(img_path)
        self.seg_path = Path(seg_path)

        self.list_img_path = list(self.img_path.glob('*.jpg'))
        self.list_seg_path = list(self.seg_path.glob('*.png'))

        self.len_img = len(self.list_img_path)
        self.len_seg = len(self.list_seg_path)

        assert self.len_img == self.len_seg, \
            f"Data is not available (Found number of {self.len_img} img and {self.len_seg} segmentation)"

        print(f"Found {self.len_img}")
        print(f"Found {self.len_seg}")

        self.transforms = transforms
        self.paired_transforms = paired_transforms

    def __len__(self):
        return self.len_img

    def __getitem__(self, item):
        img_path = self.list_img_path[item]
        seg_path = self.seg_path / img_path.name.replace('jpg', 'png')

        img = Image.open(img_path)
        seg = Image.open(seg_path)

        if self.paired_transforms is not None:
            img, seg = self.paired_transforms((img, seg))

        if self.transforms is not None:
            img = self.transforms(img)
            seg = self.transforms(seg)
        ret = {}
        ret['A'] = img
        ret['B'] = seg
        return ret





if __name__ == '__main__':

    img_path = '../data/vcoco/images/train2014'
    seg_path = '../data/vcoco/segmentation/train2014'
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

    import matplotlib.pyplot as plt

    vcoco_dataset = VCOCODataset(img_path, seg_path, transforms, paired_transforms)
    vcoco_loader = DataLoader(vcoco_dataset, shuffle=True)
    iters = iter(vcoco_loader)
    imgs = iters.next()



    print(imgs['A'].min(), imgs['A'].max())
    print(imgs['B'].min(), imgs['B'].max())

    grid = torch.cat([imgs['A'],imgs['B']], 3)
    grid = make_grid(denorm(grid))

    plt.imshow(grid.permute(1,2,0))
    plt.show()

