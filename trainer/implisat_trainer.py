import torch as T
import torch.nn.functional as F
from utils.logger import get_logger
from utils.bits import *
from torch.utils.data import Dataset
from typing import Union, Tuple, List
from config.config import Config
from trainer import BaseTrainer
from data import ImplisatDataset
from tqdm import tqdm
import pdb

class ImplisatTrainer(BaseTrainer):
    def __init__(self, model, config: Config, device: Union[str, int] = 0, log_enabled: bool = True):
        self.logger = get_logger(__class__.__name__)
        super().__init__(model, config, device, log_enabled)

    @staticmethod
    def step(model, device: int, use_checkpointing: bool, *batch_data):
        coords, pixels, channel, resolution = batch_data

        fourier_mods = model.forward_fourier(channel, resolution)
        if use_checkpointing:
            output = model.forward_inr_with_cp(coords, fourier_mods)
        else:
            output = model.forward_inr(coords, fourier_mods)

        return pixels, output

    @staticmethod
    def step_grid(model, device: int, use_checkpointing: bool, image_dim: Tuple[int, int, int], *batch_data):
        coords, pixels, channel, resolution = batch_data

        fourier_mods = model.forward_fourier(channel, resolution)
        if use_checkpointing:
            output = model.forward_inr_with_cp(coords, fourier_mods)
        else:
            output = model.forward_inr(coords, fourier_mods)

        h, w, c = image_dim

        output = output.reshape(h, w, c).permute(2, 0, 1) # (C, H, W)

        out_10 = output[:4]
        out_20 = output[4:10]
        out_60 = output[10:13]

        out_20 = F.avg_pool2d(out_20.unsqueeze(0), kernel_size=2, stride=2).squeeze(0)
        out_60 = F.avg_pool2d(out_60.unsqueeze(0), kernel_size=6, stride=6).squeeze(0)
        out_20 = F.interpolate(out_20.unsqueeze(0), size=(h, w), mode='nearest').squeeze(0)
        out_60 = F.interpolate(out_60.unsqueeze(0), size=(h, w), mode='nearest').squeeze(0)

        output = T.cat([out_10, out_20, out_60], 0)
        output = output.permute(1, 2, 0).reshape(-1, c)

        return pixels, output

    @staticmethod
    def train(
        model, 
        optim: T.optim.Optimizer, 
        loss_fn: List[T.nn.Module], 
        ds: Dataset, 
        scheduler: T.optim.lr_scheduler.LRScheduler = None, 
        device: Union[int, str] = 'cuda', 
        n_splits: int = 1,
        use_checkpointing: bool = False,
        grid_postprocess: bool = False,
        **kwargs,
    ) -> Union[float, float, T.Tensor, T.Tensor]:
        model.train()
        
        batch_data = ds[0]
        if grid_postprocess:
            pixels, out = ImplisatTrainer.step_grid(model, device, use_checkpointing, (ds.h, ds.w, ds.c), *batch_data)
        else:
            pixels, out = ImplisatTrainer.step(model, device, use_checkpointing, *batch_data)
        mse_loss = loss_fn[0](out, pixels) / n_splits # n_splits must be equal to gradient accumulation steps

        mse_loss.backward()

        optim.step()
        optim.zero_grad()
        if scheduler is not None:
            scheduler.step(mse_loss)

        gt_image = pixels.reshape(ds.h, ds.w, ds.c).permute(2, 0, 1)
        rec_image = out.reshape(ds.h, ds.w, ds.c).permute(2, 0, 1)
        
        return mse_loss.item(), -1, gt_image, rec_image

    def do_training(self, train_dataset: ImplisatDataset):
        train_dataset.move_to(self.device)
        return super().do_training(train_dataset)