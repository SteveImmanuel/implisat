import torch
import torch.nn.functional as F
import rasterio
from utils.bits import *
from PIL import Image

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, n_splits: int=1, **kwargs):
        super().__init__()
        self.path = path
        self.n_splits = n_splits

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

    def __len__(self):
        return self.n_splits

    def __getitem__(self, idx):    
        return self.coords[idx], self.pixels[idx]

class ImageCoordinateDataset(BaseDataset):
    def __init__(self, path: str, n_splits: int=1, n_bits: int=8, normalize_per_channel: bool = False, **kwargs):
        super().__init__(path, n_splits)
        self.n_bits = n_bits
        self.normalize_per_channel = normalize_per_channel

        self._load_image()
        self.coords = gen_coord_space(self.h, self.w, 8, False)
        assert self.coords.shape[0] == self.pixels.shape[0]

        self.split_size = self.coords.shape[0] // self.n_splits
        self._split()
    
    def _load_image(self):
        try:
            image = np.array(Image.open(self.path)).transpose(2, 0, 1)
        except:
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
            image = image / self.max_bits # [0, 1]

        self.pixels = torch.FloatTensor(image) # (C, H, W)
        self.pixels = self.pixels.permute(1, 2, 0).reshape(-1, self.c)