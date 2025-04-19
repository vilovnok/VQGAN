import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os
import zipfile
import requests
import subprocess
import albumentations
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image



class ImagePaths(Dataset):
    def __init__(self, path, size=None):
        self.size = size

        self.images = [os.path.join(path, file) for file in os.listdir(path)]
        self._length = len(self.images)

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def __getitem__(self, i):
        example = self.preprocess_image(self.images[i])
        return example


def load_data(args):
    train_data = ImagePaths(args.dataset_path, size=256)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    return train_loader


# --------------------------------------------- #
#                  Module Utils
#            for Encoder, Decoder etc.
# --------------------------------------------- #

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def plot_images(images):
    x = images["input"]
    reconstruction = images["rec"]
    half_sample = images["half_sample"]
    full_sample = images["full_sample"]

    fig, axarr = plt.subplots(1, 4)
    axarr[0].imshow(x.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[1].imshow(reconstruction.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[2].imshow(half_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[3].imshow(full_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    plt.show()

# --------------------------------------------- #
#              Encoder-Decoder
#               START POINT
# --------------------------------------------- #



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.block = nn.Sequential(
            GroupNorm(in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            GroupNorm(out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

        if in_channels != out_channels:
            self.channel_up = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.channel_up(x) + self.block(x)
        else:
            return x + self.block(x)


class NonLocalBlock(nn.Module):
    def __init__(self, channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = channels

        self.gn = GroupNorm(channels)
        self.k = nn.Conv2d(channels, channels, 1, 1, 0)
        self.q = nn.Conv2d(channels, channels, 1, 1, 0)
        self.v = nn.Conv2d(channels, channels, 1, 1, 0)

        self.proj_out = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        h_ = self.gn(x)
        
        q = self.q(h_)
        v = self.v(h_)
        k = self.k(h_)

        b, c, h, w = q.shape

        q = q.reshape(b, c, h*w)
        v = v.reshape(b, c, h*w)
        k = k.reshape(b, c, h*w)
        
        q = q.permute(0, 2, 1)

        attn = torch.bmm(q, k)
        attn = attn * (int(c))**(-0.5)
        attn = F.softmax(attn, dim=2)
        attn = attn.permute(0, 2, 1)
        
        A = torch.bmm(v, attn)
        A = A.reshape(b, c, h, w)

        return x + A


class UpSampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0)
        return self.conv(x)

class DownSampleBlock(nn.Module):
    def __init__(self, channels):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 2, 0)

    def forward(self, x):
        pad = (0,1,0,1)
        x = F.pad(x, pad, mode='constant', value=0)
        return self.conv(x)
    
class GroupNorm(nn.Module):
    def __init__(self, channels):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)
    

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)



# --------------------------------------------- #
#                  LPIPS
#               START POINT
# --------------------------------------------- #



URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def get_ckpt_path(name, root):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path):
        print(f"Downloading {name} model from {URL_MAP[name]} to {path}")
        download(URL_MAP[name], path)
    return path



# --------------------------------------------- #
#                  Main
#               START POINT
# --------------------------------------------- #


def download_celeba(dataset_zip="celeba-dataset.zip"):
    """Downloads the CelebA dataset using Kaggle API if it hasn't been downloaded."""
    if os.path.exists(dataset_zip):
        print(f"{dataset_zip} already exists. Skipping download.")
        return

    print("Downloading CelebA dataset from Kaggle...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", "jessicali9530/celeba-dataset"],
            check=True
        )
        print("Download complete.")
    except subprocess.CalledProcessError as e:
        print("Kaggle download failed. Make sure Kaggle API is installed and your credentials are set up.")
        raise e


def extract_dataset(dataset_zip="celeba-dataset.zip", extract_to="celeba"):
    """Extracts the CelebA dataset zip into a directory."""
    if os.path.exists(extract_to):
        print(f"Directory '{extract_to}' already exists. Skipping extraction.")
        return

    print(f"Extracting {dataset_zip} to '{extract_to}'...")
    with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")