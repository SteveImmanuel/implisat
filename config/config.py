from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class ModelConfig:
    model_type: str
    n_input_dims: int
    n_output_dims: int
    n_fourier_bases: int = 200
    n_heads: int = 1
    n_aux_dims: int = 0
    activation: str = 'ReLU'
    output_activation: str = 'Sigmoid'
    is_phase1d: bool = False
    n_neurons: int = 64
    n_layers: int = 2
    outermost_linear: bool = False
    first_omega_0: float = 30
    hidden_omega_0: float = 30

    @staticmethod
    def from_dict(config: Dict):
        return ModelConfig(**config)
    
    def to_dict(self):
        out = self.__dict__.copy()
        kept_keys = ['n_input_dims', 'n_output_dims', 'n_neurons', 'n_layers', 'model_type']
        if 'implisat' in self.model_type :
            kept_keys += ['outermost_linear', 'first_omega_0', 'hidden_omega_0', 'n_fourier_bases', 'is_phase1d', 'output_activation']

        all_keys = list(out.keys())
        for key in all_keys:
            if key not in kept_keys:
                del out[key]

        return out

@dataclass
class TrainConfig:
    epoch: int
    lr: float
    lr_decay_patience: int
    lr_decay_factor: float
    num_workers: int
    log_dir: str
    early_stop_patience: int
    model_path: str = None
    uid: str = None
    ckpt_dir: str = None
    img_log_interval: int = 10
    visualize_per_bit_plane: bool = False
    visualize_per_channel: bool = False
    visualize_max_res:int = 512
    gpu_ids:list = None
    use_checkpointing: bool = False
    grid_postprocess: bool = False
    seed: int = None

    @staticmethod
    def from_dict(config: Dict):
        return TrainConfig(**config)

    def to_dict(self):
        return self.__dict__

@dataclass
class DataConfig:
    path: str
    n_splits: int = 1
    n_bits: int = 8
    with_bit_planes: bool = True # if true, n_input_dims must be 3 
    is_sentinel: bool = False
    normalize_per_channel: bool = False
    
    @staticmethod
    def from_dict(config: Dict):
        return DataConfig(**config)
    
    def to_dict(self):
        return self.__dict__

@dataclass
class Config:
    model: ModelConfig
    train: TrainConfig
    data: DataConfig

    @staticmethod
    def from_dict(config: Dict):
        return Config(
            model=ModelConfig.from_dict(config['model']),
            train=TrainConfig.from_dict(config['train']),
            data=DataConfig.from_dict(config['data']),
        )

    def to_dict(self):
        return {
            'model': self.model.to_dict(),
            'train': self.train.to_dict(),
            'data': self.data.to_dict(),
        }