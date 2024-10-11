import torch
import torch.nn.functional as F
import rasterio
import numpy as np
from utils.bits import gen_coord_space
from PIL import Image

class ImplisatDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, n_splits: int, n_bits: int=8, normalize_per_channel: bool = False, **kwargs):
        super().__init__()
        self.n_splits = n_splits
        self.path = path
        self.n_bits = n_bits
        self.normalize_per_channel = normalize_per_channel

        self.band_order = [1, 2, 3, 7, 4, 5, 6, 8, 11, 12, 0, 9, 10]
        # self.band_order = [1, 2, 3, 7, 4, 5, 6, 10, 11, 12, 0, 8, 9]
        self._load_image()
        self._init_fourier_input()
        self.coords = gen_coord_space(self.h, self.w, with_bit_planes=False)
        assert self.coords.shape[0] == self.pixels.shape[0]
        self.split_size = self.coords.shape[0] // self.n_splits
        self._split()

    def _split(self):
        pixels_split = []
        coords_split = []
        for i in range(self.n_splits):
            if i == self.n_splits - 1:
                pixels_split.append(self.pixels[i * self.split_size:])
                coords_split.append(self.coords[i * self.split_size:])
            else:
                pixels_split.append(self.pixels[i * self.split_size:(i + 1) * self.split_size]) 
                coords_split.append(self.coords[i * self.split_size:(i + 1) * self.split_size])
        self.pixels = pixels_split
        self.coords = coords_split

    def __iter__(self):
        self._cur_index = 0
        return self
    
    def __next__(self):
        if self._cur_index < len(self):
            result = self[self._cur_index]
            self._cur_index += 1
            return result
        else:
            raise StopIteration
    
    def __len__(self):
        return self.n_splits
    
    def _init_fourier_input(self):
        self.channel = F.one_hot(torch.tensor(self.band_order), num_classes=self.c).float()
        # self.resolution = torch.tensor([0.1] * 4 + [0.2] * 6 + [0.6] * 3).unsqueeze(1)
        self.resolution = torch.tensor([1] * 4 + [2] * 6 + [6] * 3).unsqueeze(1)

    def _load_image(self):
        with rasterio.open(self.path) as src:
            image = src.read(out_shape=(src.count, src.height, src.width))

        self.c = image.shape[0]
        self.h = image.shape[1]
        self.w = image.shape[2]

        image = image.astype(np.float32)
        
        if self.normalize_per_channel:
            self.norm = []
            for i in range(image.shape[0]):
                min_val = image[i].min()
                max_val = image[i].max()
                image[i] = (image[i] - min_val) / (max_val - min_val)
                self.norm.append((min_val, max_val))
        else:
            self.max_bits = (2 ** self.n_bits) - 1
            image = image / self.max_bits

        self.pixels = torch.FloatTensor(image) # (C, H, W)
        self.pixels = self.pixels[self.band_order]
        self.pixels = self.pixels.permute(1, 2, 0).reshape(-1, self.c)
    
    def __getitem__(self, idx):    
        return self.coords[idx], self.pixels[idx], self.channel, self.resolution

    def move_to(self, device):
        self.pixels = [x.to(device) for x in self.pixels]
        self.coords = [x.to(device) for x in self.coords]
        self.channel = self.channel.to(device)
        self.resolution = self.resolution.to(device)

if __name__ == '__main__':
    ds = ImplisatDataset('images/london/patch-0-0/full.tif', normalize_per_channel=True)