import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# from data.base_dataset import BaseDataset, get_params, get_transform
# from data.image_folder import make_dataset
from PIL import Image
from pathlib import Path
import random
import numpy as np

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def get_params(size, load_size = 286, crop_size = 256):
    w, h = size
    new_h = h
    new_w = w
    new_h = new_w = load_size

    x = random.randint(0, np.maximum(0, new_w - crop_size))
    y = random.randint(0, np.maximum(0, new_h - crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(params=None, grayscale=False, method=Image.BICUBIC, convert=True, load_size=286, crop_size=256):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    osize = [load_size, load_size]

    transform_list.append(transforms.Resize(osize, method))

    transform_list.append(transforms.RandomCrop(crop_size))

    transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]
    if grayscale:
        transform_list += [transforms.Normalize((0.5,), (0.5,))]
    else:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)


class FacadeDataset(Dataset):
    """A dataset class for paired image dataset.
    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, data_path, n_crop=None, n_inc=3, n_outc=3):


        self.AB_paths = getDataInPath(data_path)
        self.n_crop = n_crop
        self.n_inc = n_inc
        self.n_outc = n_outc


    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index

        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(A.size)
        A_transform = get_transform(transform_params, grayscale=False)
        B_transform = get_transform(transform_params, grayscale=False)

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)


def getDataInPath(path : str):
    ret = list()
    path = Path(path)
    file_list = list(path.glob('*.jpg'))
    print(f"Found {len(file_list)} jpg image")
    for _data in path.glob('*.jpg'):
        # print(_data.absolute())
        ret.append(str(_data.absolute()))
    return ret


if __name__ == '__main__':
    train_dataset = FacadeDataset('../data/facade/train')
    train_loader = DataLoader(train_dataset,
                              batch_size=4)
    # _data = train_loader.dataset[10]

    # print(A)
    import matplotlib.pyplot as plt
    # plt.imshow(_data['A'].permute(1,2,0) + 1 / 2)
    # plt.show()
    # plt.imshow(_data['B'].permute(1,2,0) + 1 / 2)
    # plt.show()

    iters = iter(train_loader)
    data = iters.next()
    dataA = (data['A'] + 1) / 2

    dataB = (data['B'] + 1) / 2



    #
    from torchvision.utils import make_grid, save_image
    # plt.imshow(make_grid(dataA, nrow=1).permute(1, 2, 0))
    # plt.show()
    #
    # plt.imshow(make_grid(dataB, nrow=1).permute(1, 2, 0))
    # plt.show()

    def save_images(path, oriB, oriA, gened):
        save_path = Path(path)
        if not save_path.parent.exists():
            save_path.parent.mkdir()
        cat_data = torch.cat([oriB, oriA, gened], dim=3)
        save_image(cat_data, save_path, nrow=1)

    plt.imshow(make_grid(cat_data, nrow=1).permute(1, 2, 0))
    plt.show()