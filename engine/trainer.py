from typing import Dict, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from twostage_gan.losses.gan_losses import Pix2PixLoss


def resolve_device(device: str = 'cuda') -> torch.device:
    if device == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def train_step(input_image: torch.Tensor, target_image: torch.Tensor, generator: nn.Module, discriminator: nn.Module, generator_optimizer: Optimizer, discriminator_optimizer: Optimizer, loss_fn: Pix2PixLoss, device: torch.device) -> Dict[str, float]:
    input_image = input_image.to(device)
    target_image = target_image.to(device)

    discriminator_optimizer.zero_grad()
    with torch.no_grad():
        gen_output = generator(input_image)
    disc_real_output = discriminator(input_image, target_image)
    disc_generated_output = discriminator(input_image, gen_output.detach())
    disc_loss, _ = loss_fn.discriminator_loss(disc_real_output, disc_generated_output)
    disc_loss.backward()
    discriminator_optimizer.step()

    generator_optimizer.zero_grad()
    gen_output = generator(input_image)
    disc_generated_output_for_g = discriminator(input_image, gen_output)
    gen_total_loss, gen_gan_loss, gen_l1_loss, _ = loss_fn.generator_loss(disc_generated_output_for_g, gen_output, target_image)
    gen_total_loss.backward()
    generator_optimizer.step()

    return {
        'gen_total_loss': gen_total_loss.item(),
        'gen_gan_loss': gen_gan_loss.item(),
        'gen_l1_loss': gen_l1_loss.item(),
        'disc_loss': disc_loss.item(),
    }


def train_one_stage(train_loader, generator: nn.Module, discriminator: nn.Module, generator_optimizer: Optimizer, discriminator_optimizer: Optimizer, loss_fn: Pix2PixLoss, device: torch.device, epochs: int, source_index: int, target_index: int, log_interval: int = 50, stage_name: str = 'stage'):
    history = []
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        for step, batch in enumerate(train_loader, start=1):
            losses = train_step(batch[source_index], batch[target_index], generator, discriminator, generator_optimizer, discriminator_optimizer, loss_fn, device)
            history.append(losses)
            if step % log_interval == 0:
                print(f"[{stage_name}] Epoch {epoch + 1}, Step {step}, Gen Loss: {losses['gen_total_loss']:.4f}, Disc Loss: {losses['disc_loss']:.4f}")
    return history


def build_optimizers(generator: nn.Module, discriminator: nn.Module, generator_lr: float, discriminator_lr: float, beta1: float, beta2: float) -> Tuple[Optimizer, Optimizer]:
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=generator_lr, betas=(beta1, beta2))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=discriminator_lr, betas=(beta1, beta2))
    return generator_optimizer, discriminator_optimizer
