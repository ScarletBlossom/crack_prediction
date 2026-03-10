import argparse

from twostage_gan.config import DataConfig, InferenceConfig, TrainConfig
from twostage_gan.datasets.triplet_dataset import get_dataloaders
from twostage_gan.engine.trainer import resolve_device
from twostage_gan.models.generator import Generator
from twostage_gan.utils.checkpoint import load_weights
from twostage_gan.utils.visualization import inference_and_visualize_5cols


def parse_args():
    parser = argparse.ArgumentParser(description='Run two-stage inference.')
    parser.add_argument('--train-dir', default=DataConfig.train_dir)
    parser.add_argument('--test-dir', default=DataConfig.test_dir)
    parser.add_argument('--batch-size', type=int, default=DataConfig.batch_size)
    parser.add_argument('--num-workers', type=int, default=DataConfig.num_workers)
    parser.add_argument('--generator1-path', default='./checkpoints/generator1_sed.pth')
    parser.add_argument('--generator2-path', default='./checkpoints/generator2_crack.pth')
    parser.add_argument('--save-dir', default=InferenceConfig.output_dir)
    parser.add_argument('--device', default=TrainConfig.device)
    parser.add_argument('--max-images', type=int, default=InferenceConfig.max_images_per_batch)
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    generator1 = load_weights(Generator().to(device), args.generator1_path, map_location=device)
    generator2 = load_weights(Generator().to(device), args.generator2_path, map_location=device)

    _, test_loader = get_dataloaders(args.train_dir, args.test_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    inference_and_visualize_5cols(generator1, generator2, test_loader, device, save_dir=args.save_dir, max_images=args.max_images)
    print(f'Inference results saved to: {args.save_dir}')


if __name__ == '__main__':
    main()
