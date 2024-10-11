import torch
import numpy as np
from typing import Union

def bit_decomposition(image: np.ndarray, n_bits: int=8, base: int=2) -> torch.Tensor:
    # image (C, H, W) raw unnormalized
    image = np.clip(image, 0, base ** n_bits - 1)
    image = torch.LongTensor(image).int()
    res = []
    for _ in range(n_bits):
        res.append(image % base)
        image = image // base
    res = torch.stack(res, dim=0)
    res = res.to(torch.bool)
    return res

def gen_coord_space(height: int, width: int, n_bits: int=16, with_bit_planes: bool=True):
    tensors_x = torch.linspace(-1, 1, steps=height)
    tensors_y = torch.linspace(-1, 1, steps=width)
    if with_bit_planes:
        bit_plane_info = torch.linspace(-1, 1, steps=n_bits)
        tensors = (tensors_x, tensors_y, bit_plane_info)
    else:
        tensors = (tensors_x, tensors_y)

    meshgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
    meshgrid = meshgrid.reshape(-1, 3 if with_bit_planes else 2)
    return meshgrid

def calc_psnr(image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
    # assumes image1 and image2 in range [0, 1]
    mse = torch.mean((image1 - image2) ** 2)
    return (-10 * torch.log10(mse)).item()

def img_from_bit_planes(bit_planes: torch.Tensor, n_bits: int, n_channels: int, h: int, w: int, permute: bool = False) -> torch.Tensor:
    bit_multiplier = 2 ** torch.arange(n_bits)
    bit_planes = bit_planes.round()
    
    if permute:
        bit_planes = bit_planes.reshape(-1, n_channels, n_bits).permute(0, 2, 1)
    else:
        bit_planes = bit_planes.reshape(-1, n_bits, n_channels)
        
    out = (bit_planes * bit_multiplier[None, :, None]).sum(dim=1) / (2 ** n_bits - 1)
    out = out.reshape(h, w, n_channels).permute(2, 0, 1)
    return out # range [0, 1] (C, H, W)

def img_bits_from_bit_planes(bit_planes: torch.Tensor, n_bits: int, n_channels: int, h: int, w: int) -> torch.Tensor:
    bit_multiplier = 2 ** torch.arange(n_bits)
    bit_planes = bit_planes.round().reshape(-1, n_bits, n_channels)
    out = (bit_planes * bit_multiplier[None, :, None]).sum(dim=1) / (2 ** n_bits - 1)
    out = out.reshape(h, w, n_channels).permute(2, 0, 1)
    img_bits = bit_planes.reshape(h, w, n_bits, n_channels).permute(2, 3, 0, 1)
    return out, img_bits # range [0, 1] (n_bits, C, H, W)