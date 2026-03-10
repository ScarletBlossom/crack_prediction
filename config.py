from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    train_dir: str = './train_img/'
    test_dir: str = './test_img/'
    image_width: int = 256
    image_height: int = 256
    resize_height: int = 286
    resize_width: int = 286
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class TrainConfig:
    epochs: int = 100
    generator_lr: float = 1e-4
    discriminator_lr: float = 1e-5
    beta1: float = 0.5
    beta2: float = 0.999
    lambda_l1: float = 100.0
    log_interval: int = 50
    device: str = 'cuda'
    output_dir: str = './checkpoints'


@dataclass
class InferenceConfig:
    output_dir: str = './outputs_test_5cols'
    max_images_per_batch: int = 10


PROJECT_ROOT = Path(__file__).resolve().parent
