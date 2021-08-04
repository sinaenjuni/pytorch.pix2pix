
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


from PIL import Image
import numpy as np
import cv2


class VCOCOdataset(Dataset):
    def __init__(self, img_path, seg_path, transforms=None):
        self.img_path = Path(img_path)
        self.seg_path = Path(seg_path)

        self.list_img_path = list(self.img_path.glob('*.jpg'))
        self.list_seg_path = list(self.seg_path.glob('*.npy'))

        self.len_img = len(self.list_img_path)
        self.len_seg = len(self.list_seg_path)

        assert self.len_img == self.len_seg, f"Data is not available (Found number of {self.len_img} img and {self.len_seg} segmentation)"

        print(f"Found {self.len_img}")
        print(f"Found {self.len_seg}")

        self.transforms = transforms


    def __len__(self):
        return self.len_img


    def __getitem__(self, item):
        img_path = self.list_img_path[item]
        seg_path = self.seg_path / img_path.name.replace('jpg', 'npy')

        img = Image.open(img_path)
        seg = Image.open(seg_path)


        # print(img)
        # print(seg)
        # seg_path = self.
        if self.transforms is not None:
            img = self.transforms(img)
            seg = self.transforms(seg)

        return img, seg






if __name__ == '__main__':

    img_path = '../data/vcoco/images/train2014'
    seg_path = '../data/vcoco/segmentation/train2014'
    transforms = T.Compose([
                            T.Resize((256, 256)),
                            T.ToTensor(),
                            # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            T.Normalize((0.5), (0.5)),
                            T.RandomHorizontalFlip(0.5),
                            ])




    import matplotlib.pyplot as plt

    hico_dataset = VCOCOdataset(img_path, seg_path, transforms)
    img, seg = hico_dataset[1]

    plt.imshow(img.permute(1,2,0))
    plt.show()