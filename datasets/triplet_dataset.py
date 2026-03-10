import glob
import os
import random
from dataclasses import dataclass
from typing import Callable, Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np


@dataclass
class TripletFormat:
    total_parts: int
    input_index: int
    middle_index: int
    target_index: int


TRAIN_FORMAT = TripletFormat(total_parts=3, input_index=0, middle_index=1, target_index=2)
TEST_FORMAT = TripletFormat(total_parts=5, input_index=0, middle_index=1, target_index=3)



def to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[:, :, None]
    return torch.from_numpy(array).permute(2, 0, 1)


def normalize_to_minus1_1(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor - 0.5) * 2.0


def resize(image: Image.Image, height: int, width: int) -> Image.Image:
    return image.resize((width, height), Image.NEAREST)


def split_triplet_image(image_path: str, layout: TripletFormat) -> Tuple[Image.Image, Image.Image, Image.Image]:
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    single_width = width // layout.total_parts

    def crop(part_index: int) -> Image.Image:
        left = part_index * single_width
        right = (part_index + 1) * single_width
        return image.crop((left, 0, right, height))

    return crop(layout.input_index), crop(layout.middle_index), crop(layout.target_index)


def random_crop(
    input_image: Image.Image,
    middle_image: Image.Image,
    target_image: Image.Image,
    crop_h: int = 256,
    crop_w: int = 256,
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    width, height = input_image.size
    if width == crop_w and height == crop_h:
        return input_image, middle_image, target_image

    x = random.randint(0, width - crop_w)
    y = random.randint(0, height - crop_h)
    return (
        input_image.crop((x, y, x + crop_w, y + crop_h)),
        middle_image.crop((x, y, x + crop_w, y + crop_h)),
        target_image.crop((x, y, x + crop_w, y + crop_h)),
    )


def random_jitter(
    input_image: Image.Image,
    middle_image: Image.Image,
    target_image: Image.Image,
    resize_h: int = 286,
    resize_w: int = 286,
    crop_h: int = 256,
    crop_w: int = 256,
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    input_image = resize(input_image, resize_h, resize_w)
    middle_image = resize(middle_image, resize_h, resize_w)
    target_image = resize(target_image, resize_h, resize_w)
    input_image, middle_image, target_image = random_crop(input_image, middle_image, target_image, crop_h, crop_w)

    if random.random() > 0.5:
        input_image = input_image.transpose(Image.FLIP_LEFT_RIGHT)
        middle_image = middle_image.transpose(Image.FLIP_LEFT_RIGHT)
        target_image = target_image.transpose(Image.FLIP_LEFT_RIGHT)
    return input_image, middle_image, target_image


class TripletImageDataset(Dataset):
    def __init__(self, img_dir: str, pattern: str, layout: TripletFormat, transform_fn: Callable):
        self.files = sorted(glob.glob(os.path.join(img_dir, pattern)))
        self.layout = layout
        self.transform_fn = transform_fn

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        image_path = self.files[idx]
        input_image, middle_image, target_image = split_triplet_image(image_path, self.layout)
        input_image, middle_image, target_image = self.transform_fn(input_image, middle_image, target_image)
        return (
            normalize_to_minus1_1(to_tensor(input_image)),
            normalize_to_minus1_1(to_tensor(middle_image)),
            normalize_to_minus1_1(to_tensor(target_image)),
        )


class TrainDataset(TripletImageDataset):
    def __init__(self, img_dir: str):
        super().__init__(img_dir=img_dir, pattern='Train_*.jpg', layout=TRAIN_FORMAT, transform_fn=random_jitter)


class TestDataset(TripletImageDataset):
    def __init__(self, img_dir: str, image_height: int = 256, image_width: int = 256):
        super().__init__(
            img_dir=img_dir,
            pattern='Test_*.jpg',
            layout=TEST_FORMAT,
            transform_fn=lambda a, b, c: (
                resize(a, image_height, image_width),
                resize(b, image_height, image_width),
                resize(c, image_height, image_width),
            ),
        )


def get_dataloaders(train_path: str, test_path: str, batch_size: int = 16, num_workers: int = 4, pin_memory: bool = True):
    train_dataset = TrainDataset(train_path)
    test_dataset = TestDataset(test_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader
