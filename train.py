import argparse
from pathlib import Path

from twostage_gan.config import DataConfig, TrainConfig
from twostage_gan.datasets.triplet_dataset import get_dataloaders
from twostage_gan.engine.trainer import build_optimizers, resolve_device, train_one_stage
from twostage_gan.losses.gan_losses import Pix2PixLoss
from twostage_gan.models.discriminator import Discriminator
from twostage_gan.models.generator import Generator
from twostage_gan.utils.checkpoint import save_weights


def parse_args():
    parser = argparse.ArgumentParser(description='Train stage-1 or stage-2 models.')
    parser.add_argument('--stage', choices=['stage1', 'stage2'], required=True)
    parser.add_argument('--train-dir', default=DataConfig.train_dir)
    parser.add_argument('--test-dir', default=DataConfig.test_dir)
    parser.add_argument('--epochs', type=int, default=TrainConfig.epochs)
    parser.add_argument('--batch-size', type=int, default=DataConfig.batch_size)
    parser.add_argument('--num-workers', type=int, default=DataConfig.num_workers)
    parser.add_argument('--generator-lr', type=float, default=TrainConfig.generator_lr)
    parser.add_argument('--discriminator-lr', type=float, default=TrainConfig.discriminator_lr)
    parser.add_argument('--lambda-l1', type=float, default=TrainConfig.lambda_l1)
    parser.add_argument('--log-interval', type=int, default=TrainConfig.log_interval)
    parser.add_argument('--output-dir', default=TrainConfig.output_dir)
    parser.add_argument('--device', default=TrainConfig.device)
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    train_loader, _ = get_dataloaders(args.train_dir, args.test_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator_optimizer, discriminator_optimizer = build_optimizers(generator, discriminator, args.generator_lr, args.discriminator_lr, TrainConfig.beta1, TrainConfig.beta2)
    loss_fn = Pix2PixLoss(lambda_l1=args.lambda_l1)

    if args.stage == 'stage1':
        source_index, target_index = 0, 1
        generator_name, discriminator_name, stage_name = 'generator1_sed.pth', 'discriminator1_sed.pth', 'Step1'
    else:
        source_index, target_index = 1, 2
        generator_name, discriminator_name, stage_name = 'generator2_crack.pth', 'discriminator2_crack.pth', 'Step2'

    train_one_stage(train_loader, generator, discriminator, generator_optimizer, discriminator_optimizer, loss_fn, device, args.epochs, source_index, target_index, args.log_interval, stage_name)

    output_dir = Path(args.output_dir)
    save_weights(generator, str(output_dir / generator_name))
    save_weights(discriminator, str(output_dir / discriminator_name))
    print(f'Saved weights to: {output_dir.resolve()}')


if __name__ == '__main__':
    main()
