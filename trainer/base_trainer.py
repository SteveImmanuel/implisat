from abc import abstractmethod
import time
import os
import torch as T
import yaml
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List
from tqdm import tqdm
from utils.logger import get_logger
from utils.bits import *
from torch.utils.data import Dataset
from typing import Union, Tuple
from config.config import Config
from torch.optim.lr_scheduler import ReduceLROnPlateau

@dataclass
class Tracker:
    last_loss: float = None
    last_metric: float  = None
    epoch: int = 0
    step_counter: int = 0
    best_epoch: int = None
    best_metric: float = None
    direction: str = 'max'

    def inc_step_counter(self):
        self.step_counter += 1
    
    def is_metric_better(self, epoch=None):
        def _compare(a, b):
            return a > b if self.direction == 'max' else a < b

        if self.best_metric is None or _compare(self.last_metric, self.best_metric):
            self.best_metric = self.last_metric
            self.best_epoch = epoch
            return True
        return False

class BaseTrainer:
    def __init__(self, model: T.nn.Module, config: Config, device: Union[str, int] = 0, log_enabled: bool = True):
        self.logger = get_logger(__class__.__name__) if self.logger is None else self.logger
        self.model = model.to(device)
        self.config = config
        self.log_enabled = log_enabled
        self.device = device

        if config.train.uid is not None:
            self.uid = config.train.uid
        else:
            self.uid = int(time.time())

        self.loss_fn = self.get_loss_fn()
        self.optim = self._get_optimizer()
        self.scheduler = self._get_scheduler()

        if log_enabled:
            self.config.train.log_dir = os.path.join(self.config.train.log_dir, f'{self.uid}')
            self.summary_writer = SummaryWriter(log_dir=self.config.train.log_dir)
            self.config.train.ckpt_dir = os.path.join(self.config.train.log_dir, 'weights')
            os.makedirs(self.config.train.ckpt_dir, exist_ok=True)
            self.save_config()

        self.tracker = Tracker()
    
    @staticmethod
    def get_loss_fn(**kwargs) -> T.nn.Module:
        return [T.nn.MSELoss(), T.nn.L1Loss()]
    
    def _get_optimizer(self) -> T.optim.Optimizer:
        return T.optim.Adam(lr=self.config.train.lr, params=self.model.parameters(), betas=(0.9, 0.999), eps=1e-15)
    
    def _get_scheduler(self) -> T.optim.lr_scheduler._LRScheduler:
        return ReduceLROnPlateau(self.optim, mode='min', factor=self.config.train.lr_decay_factor, patience=self.config.train.lr_decay_patience)

    def save_config(self):
        config = self.config.to_dict()

        self.logger.info('======CONFIGURATIONS======')
        for k in config:
            self.logger.info(f'{k.upper()}')
            v = config[k]
            for ik, iv in v.items():
                self.logger.info(f'\t{ik.upper()}: {iv}')
        
        config_path = os.path.join(self.config.train.log_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        self.logger.info(f'Training config saved to {config_path}')

    def save_checkpoint(self, epoch: int, name: str = '', only_model: bool = True):
        save_checkpoint = {'model': self.model.state_dict()}
        if not only_model:
            save_checkpoint['optimizer'] = self.optim.state_dict()
        if name != '':
            ckpt_path = os.path.join(self.config.train.ckpt_dir, f'{name}.pt')
        else:
            ckpt_path = os.path.join(
                self.config.train.ckpt_dir,
                f'epoch{epoch:02}_loss{self.tracker.last_loss:.4f}_metric{self.tracker.last_metric:.4f}.pt',
            )
        T.save(save_checkpoint, ckpt_path)
    
    def load_checkpoint(self, ckpt_path: str):
        assert os.path.exists(ckpt_path)
        checkpoint = T.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint:
            self.optim.load_state_dict(checkpoint['optimizer'])
        self.logger.info(f'Succesfully loaded model in {ckpt_path}')
    
    @staticmethod
    def step(model: T.nn.Module, device: int, *batch_data):
        coords = batch_data[0].to(device)
        pixels = batch_data[1].to(device)

        coords = coords.reshape(-1, coords.shape[-1])
        pixels = pixels.reshape(-1, pixels.shape[-1])

        output = model(coords)
        pixels = pixels.to(output.dtype)

        return pixels, output

    @staticmethod
    @abstractmethod
    def train(
        model: T.nn.Module, 
        optim: T.optim.Optimizer, 
        loss_fn: List[T.nn.Module], 
        ds: Dataset, 
        scheduler: T.optim.lr_scheduler.LRScheduler = None, 
        device: Union[int, str] = 'cuda', 
        n_splits: int = 1,
        **kwargs,
    ) -> Tuple[float, float, T.Tensor, T.Tensor]:
        raise NotImplementedError()

    def do_training(self, train_dataset: Dataset):
        epoch = self.config.train.epoch
        early_stop_patience = self.config.train.early_stop_patience
        early_stop = False
        pbar = tqdm(range(epoch))

        for i in pbar:
            out = self.train(
                self.model, 
                self.optim, 
                self.loss_fn, 
                train_dataset, 
                self.scheduler, 
                self.device, 
                self.config.data.n_splits, 
                visualize_per_bit_plane=self.config.train.visualize_per_bit_plane,
                use_checkpointing=self.config.train.use_checkpointing,
                grid_postprocess=self.config.train.grid_postprocess,
            )

            if len(out) == 4:
                loss, l1_loss, gt_image, rec_image = out
                gt_image_bits = rec_image_bits = None
            else:
                loss, l1_loss, gt_image, rec_image, gt_image_bits, rec_image_bits = out

            psnr = calc_psnr(gt_image, rec_image)
            self.summary_writer.add_scalar('PSNR', psnr, i)
            self.summary_writer.add_scalar('Loss', loss, i)
            self.summary_writer.add_scalar('BER', l1_loss, i)
            self.summary_writer.add_scalar('Learning Rate', self.optim.param_groups[0]['lr'], i)

            self.tracker.last_metric = psnr
            self.tracker.last_loss = loss

            if i % self.config.train.img_log_interval == 0:
                self.visualize(gt_image, rec_image, i, gt_image_bits=gt_image_bits, rec_image_bits=rec_image_bits)

            pbar.set_postfix({
                'Loss': loss,
                'BER': l1_loss, # Bit error rate
                'PSNR': psnr,
                'LR': self.optim.param_groups[0]['lr'],
            })
        
            if self.tracker.is_metric_better(i + 1):
                self.save_checkpoint(i + 1, 'best')
            elif early_stop_patience > 0 and i + 1 - self.tracker.best_epoch > early_stop_patience:
                early_stop = True
            
            if l1_loss == 0:
                self.logger.info('Perfect reconstruction achieved. Stopping training.')
                self.save_checkpoint(i + 1, 'perfect')
                break

            if early_stop:
                self.logger.info(f'Early stopping. No improvement in metric for the last {early_stop_patience} epochs.')
                break
        self.logger.info(f'Best result was seen in epoch {self.tracker.best_epoch}')
        return self.tracker.best_metric

    def visualize(self, gt_img: T.Tensor, rec_img: T.Tensor, epoch: int, **kwargs):
        # assumes img in range [0, 1] with shape (C, H, W)
        error_map = T.abs(gt_img - rec_img)
        error_map = (error_map - error_map.min()) * error_map.max() / (error_map.max() - error_map.min())
        vis_img = T.cat([gt_img, rec_img, error_map], dim=2)
        
        if vis_img.shape[0] > 3:
            if not self.config.train.visualize_per_channel:
                out = vis_img[[3, 2, 1]] # RGB band 4, 3, 2
                out = (out * 20).clamp(0, 1)
                self.summary_writer.add_image('Reconstruction RGB', out, epoch, dataformats='CHW')

                out = vis_img[[7, 3, 2]] # False color band 8, 4, 3
                out = (out * 20).clamp(0, 1)
                self.summary_writer.add_image('Reconstruction False Color', out, epoch, dataformats='CHW')

                out = vis_img[[12, 7, 3]] # SWIR band 12, 8, 4
                out = (out * 20).clamp(0, 1)
                self.summary_writer.add_image('Reconstruction SWIR', out, epoch, dataformats='CHW')
            else:
                gt_all = []
                rec_all = []
                error_all = []
                for i in range(vis_img.shape[0]):
                    gt_all.append(gt_img[i])
                    rec_all.append(rec_img[i])
                    error_map = T.abs(gt_img[i] - rec_img[i])
                    error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min())
                    error_all.append(error_map)

                gt_all = T.cat(gt_all, dim=0)
                rec_all = T.cat(rec_all, dim=0)
                error_all = T.cat(error_all, dim=0)
                out = T.cat([gt_all, rec_all, error_all], dim=1)
                out = out.clamp(0, 1)
                self.summary_writer.add_image(f'Reconstruction Per Channel', out, epoch, dataformats='HW')
        else:
            title = 'Reconstruction'
            if kwargs.get('title', None) is not None:
                title = kwargs['title']
            self.summary_writer.add_image(title, vis_img, epoch, dataformats='CHW')

    