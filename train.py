import argparse
import torch as T
import yaml
import random
import numpy as np
import pdb
from typing import Tuple, Type
from utils.logger import *
from data import *
from model import *
from trainer import *
from config.config import Config

def setup_factory(config: Config) -> Tuple[Type[BaseTrainer], Type[BaseDataset], Type[T.nn.Module]]:
    model_factory = Implisat
    trainer_factory = ImplisatTrainer
    dataset_factory = ImplisatDataset
    
    return trainer_factory, dataset_factory, model_factory

def main(config: Config):
    setup_logging()
    logger = get_logger(__name__)

    trainer_factory, dataset_factory, model_factory = setup_factory(config)
    logger.info('Preparing dataset')
    train_dataset = dataset_factory(**config.data.to_dict())
    logger.info(f'Training for image with shape {train_dataset.h}x{train_dataset.w}x{train_dataset.c}')

    logger.info('Instantiating model')
    model = model_factory(**config.model.to_dict())
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Num of parameters: {total_params}')
    
    logger.info('Preparing trainer agent')
    trainer: BaseTrainer = trainer_factory(model, config, 'cuda' if T.cuda.is_available() else T.device('cpu'))
    if config.train.model_path is not None:
        trainer.load_checkpoint(config.train.model_path)
    
    best_metric = trainer.do_training(train_dataset)
    return best_metric, total_params

def get_args():
    parser = argparse.ArgumentParser('INR Image Compression', add_help=False)
    parser.add_argument('--uid', type=str, help='unique id for the run', default=None)
    parser.add_argument('--config', type=str, help='path to yaml config', default='config/base.yaml')
    parser.add_argument('--model-path', type=str, help='ckpt path to continue', default=None)
    parser.add_argument('--img', type=str, help='img path', default=None)
    parser.add_argument('--patience', type=int, help='patience for early stopping', default=-1)
    parser.add_argument('--seed', type=int, help='random seed', default=None)
    return parser.parse_args()

def parse_gpu_ids(gpu_ids_str):
    return [int(x) for x in gpu_ids_str.split(',')]

def seed_everything(seed: int):    
    random.seed(seed)
    np.random.seed(seed)
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = True

if __name__ == '__main__':
    args = get_args()
    train_args = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    config = Config.from_dict(train_args)
    if args.seed is None and config.train.seed is None:
        config.train.seed = random.randint(0, 1000000)
    seed_everything(config.train.seed)

    if args.uid is not None:
        config.train.uid = args.uid
    if args.model_path is not None:
        config.train.model_path = args.model_path
    if args.patience != -1:
        config.train.early_stop_patience = args.patience
    if args.img is not None:
        config.data.path = args.img

    main(config)