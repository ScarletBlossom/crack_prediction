from pathlib import Path

import torch
import torch.nn as nn


def save_weights(model: nn.Module, save_path: str) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_weights(model: nn.Module, ckpt_path: str, map_location=None) -> nn.Module:
    state_dict = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(state_dict)
    return model
