import torch
import torch.nn as nn


class Pix2PixLoss:
    def __init__(self, lambda_l1: float = 100.0):
        self.lambda_l1 = lambda_l1
        self.adv_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()

    def generator_loss(self, disc_generated_output: torch.Tensor, gen_output: torch.Tensor, target: torch.Tensor):
        gan_loss = self.adv_loss(disc_generated_output, torch.ones_like(disc_generated_output))
        l1_loss = self.l1_loss(gen_output, target)
        total_gen_loss = gan_loss + self.lambda_l1 * l1_loss
        return total_gen_loss, gan_loss, l1_loss, l1_loss

    def discriminator_loss(self, disc_real_output: torch.Tensor, disc_generated_output: torch.Tensor):
        real_loss = self.adv_loss(disc_real_output, torch.ones_like(disc_real_output))
        fake_loss = self.adv_loss(disc_generated_output, torch.zeros_like(disc_generated_output))
        total_disc_loss = real_loss + fake_loss
        return total_disc_loss, real_loss
