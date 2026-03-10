import os

import matplotlib.pyplot as plt
import torch


def _denormalize(image_tensor: torch.Tensor) -> torch.Tensor:
    return (image_tensor * 0.5 + 0.5).clamp(0, 1)


def inference_and_visualize_5cols(generator1, generator2, test_loader, device, save_dir='./outputs_test_5cols', max_images=10):
    os.makedirs(save_dir, exist_ok=True)
    generator1.eval()
    generator2.eval()

    with torch.no_grad():
        for batch_idx, (input_image, sed_gt, crack_gt) in enumerate(test_loader):
            input_image = input_image.to(device)
            sed_gt = sed_gt.to(device)
            crack_gt = crack_gt.to(device)
            sed_pred = generator1(input_image)
            crack_pred = generator2(sed_pred)

            num_images = min(max_images, input_image.size(0))
            for image_idx in range(num_images):
                fig, axes = plt.subplots(1, 5, figsize=(25, 5))
                panels = [
                    ('Input Image', input_image[image_idx]),
                    ('SED Ground Truth', sed_gt[image_idx]),
                    ('SED Prediction', sed_pred[image_idx]),
                    ('Crack Ground Truth', crack_gt[image_idx]),
                    ('Crack Prediction', crack_pred[image_idx]),
                ]
                for ax, (title, tensor) in zip(axes, panels):
                    ax.imshow(_denormalize(tensor).cpu().permute(1, 2, 0))
                    ax.set_title(title)
                    ax.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'sample_{batch_idx}_{image_idx}.png'), dpi=300, transparent=True)
                plt.close(fig)


def inference_and_visualize_3cols(generator, test_loader, device, save_dir='./outputs_test_3cols', max_images=10):
    os.makedirs(save_dir, exist_ok=True)
    generator.eval()

    with torch.no_grad():
        for batch_idx, (_, input_image, sed_gt) in enumerate(test_loader):
            input_image = input_image.to(device)
            sed_gt = sed_gt.to(device)
            sed_pred = generator(input_image)

            num_images = min(max_images, input_image.size(0))
            for image_idx in range(num_images):
                fig, axes = plt.subplots(1, 3, figsize=(25, 5))
                panels = [('Input Image', input_image[image_idx]), ('SED Ground Truth', sed_gt[image_idx]), ('SED Prediction', sed_pred[image_idx])]
                for ax, (title, tensor) in zip(axes, panels):
                    ax.imshow(_denormalize(tensor).cpu().permute(1, 2, 0))
                    ax.set_title(title)
                    ax.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'sample_{batch_idx}_{image_idx}.png'), dpi=300, transparent=True)
                plt.close(fig)
            break
