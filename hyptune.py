import optuna
import os
import argparse
from optuna.trial import BaseTrial
from train import main, seed_everything
import time
import yaml
import random
from config.config import Config

def wrapper_objective(img_path: str, config_path: str):
    def objective(trial: BaseTrial):
        seed = random.randint(0, 1000000)
        seed_everything(seed)
        train_args = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
        config = Config.from_dict(train_args)
        config.train.seed = seed
        
        config.train.uid = int(time.time())
        config.data.path = img_path
        log_name = os.path.basename(img_path).split('.')[0]
        config.train.log_dir = f'logs/hyptune_{log_name}'
        
        config.model.n_neurons = trial.suggest_categorical('n_neurons', [128, 256])
        config.model.n_fourier_bases = trial.suggest_categorical('n_fourier_bases', [256, 1024, 4096])
        config.model.n_layers = trial.suggest_int('n_layers', 2, 4)
        config.model.is_phase1d = trial.suggest_categorical('is_phase1d', [True, False])
        config.model.outermost_linear = trial.suggest_categorical('outermost_linear', [True, False])
        config.model.output_activation = trial.suggest_categorical('output_activation', ['Sigmoid', 'None'])
        config.train.lr = trial.suggest_categorical('lr', [1e-4, 2.5e-4, 5e-4, 1e-3, 5e-3])

        psnr, total_params = main(config)

        trial.set_user_attr('uid', config.train.uid)
        trial.set_user_attr('seed', seed)
        return psnr, total_params
    return objective

def get_args():
    parser = argparse.ArgumentParser('INR Image Compression - Hyperparameter Tuning', add_help=False)
    parser.add_argument('--img', type=str, help='img path', default=None)
    parser.add_argument('--config', type=str, help='path to yaml config', default='config/implisat.yaml')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    study_name = os.path.basename(args.img).split('.')[0]
    storage_name = f'sqlite:///hyptune_{study_name}.db'
    study = optuna.create_study(study_name=study_name, storage=storage_name, directions=['maximize', 'minimize'], load_if_exists=True)
    study.set_metric_names(['PSNR', 'Total Params'])
    study.optimize(wrapper_objective(args.img, args.config))
