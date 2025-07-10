import os
import glob
import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms import functional as F
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_msssim import ssim

class SSIMLoss(nn.Module):
    def forward(self, x, y):
        return 1 - ssim(x, y, data_range=1.0, size_average=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform
image_transforms = transforms.Compose([
    transforms.Resize((128, 128)),  # match model input
    transforms.ToTensor(),
])

class RICECloudDataset(Dataset):
    def __init__(self, cloud_dir, label_dir, transform=None):
        self.cloud_paths = sorted(glob.glob(os.path.join(cloud_dir, "*.png")))
        self.label_paths = sorted(glob.glob(os.path.join(label_dir, "*.png")))
        assert len(self.cloud_paths) > 0, f"No images found in {cloud_dir}"
        assert len(self.cloud_paths) == len(self.label_paths), "Mismatch in image count"
        self.transform = transform

    def __len__(self):
        return len(self.cloud_paths)

    def __getitem__(self, idx):
        cloud_image = Image.open(self.cloud_paths[idx]).convert("RGB")
        label_image = Image.open(self.label_paths[idx]).convert("RGB")

        if self.transform:
            cloud_image = self.transform(cloud_image)
            label_image = self.transform(label_image)

        return cloud_image, label_image


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb_scale = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = time[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)  # <-- inserted SEBlock here
        self.act2 = nn.SiLU()

    def forward(self, x, t):
        x = self.conv1(x)
        x = self.bn1(x)

        t_emb = self.time_mlp(t)  # shape [B, out_ch]
        t_emb = t_emb[:, :, None, None]  # reshape to [B, out_ch, 1, 1]

        x = x + t_emb
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.se(x)       # <-- apply SE attention here
        x = self.act2(x)

        return x


class CloudDenoiser(nn.Module):
    def __init__(self, in_channels=3, base_dim=64):
        super().__init__()
        self.time_mlp = SinusoidalPositionEmbeddings(base_dim)

        self.down1 = UNetBlock(in_channels, base_dim, base_dim)
        self.down2 = UNetBlock(base_dim, base_dim * 2, base_dim)
        self.down3 = UNetBlock(base_dim * 2, base_dim * 4, base_dim)

        self.bottleneck = UNetBlock(base_dim * 4, base_dim * 8, base_dim)

        self.up1 = UNetBlock(base_dim * 8 + base_dim * 4, base_dim * 4, base_dim)
        self.up2 = UNetBlock(base_dim * 4 + base_dim * 2, base_dim * 2, base_dim)
        self.up3 = UNetBlock(base_dim * 2 + base_dim, base_dim, base_dim)

        self.final_conv = nn.Conv2d(base_dim, in_channels, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)

        # Encoder
        x1 = self.down1(x, t)
        x2 = self.down2(nn.MaxPool2d(2)(x1), t)
        x3 = self.down3(nn.MaxPool2d(2)(x2), t)

        # Bottleneck
        x4 = self.bottleneck(nn.MaxPool2d(2)(x3), t)

        # Decoder
        x = nn.Upsample(scale_factor=2, mode='nearest')(x4)
        x = self.up1(torch.cat([x, x3], dim=1), t)

        x = nn.Upsample(scale_factor=2, mode='nearest')(x)
        x = self.up2(torch.cat([x, x2], dim=1), t)

        x = nn.Upsample(scale_factor=2, mode='nearest')(x)
        x = self.up3(torch.cat([x, x1], dim=1), t)

        return self.final_conv(x)


class DiffusionProcess:
    def __init__(self, T=1000):
        self.T = T


def evaluate_metrics(pred, target):
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    return {
        'PSNR': psnr(pred, target, data_range=1.0).item(),
        'SSIM': ssim(pred, target, data_range=1.0).item()
    }

def visualize_outputs(noisy, pred, target, epoch, save_dir='visuals_res'):
    os.makedirs(save_dir, exist_ok=True)
    idx = 0  # just show the first image in batch
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    for ax, img, title in zip(axs, [noisy, pred, target], ['Cloudy', 'Denoised', 'Ground Truth']):
        img = img[idx].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/epoch_{epoch}.png')
    plt.close()


def train(model, diffuser, train_loader, val_loader, num_epochs=50, save_path='residual_best_model.pth'):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_psnr = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0

        for noisy_images, gt_images in train_loader:
            noisy_images = noisy_images.to(device)
            gt_images = gt_images.to(device)
            t = torch.randint(0, diffuser.T, (noisy_images.size(0),), device=device)

            # Predict noise = (noisy - clean)
            pred_residual = model(noisy_images, t)
            denoised = (noisy_images - pred_residual).clamp(0, 1)
            mse = nn.MSELoss()(denoised, gt_images)
            ssim_value = SSIMLoss()(denoised, gt_images)
            loss = 0.8 * mse + 0.2 * ssim_value


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print('----------------------------------------------')
        print(f"[Epoch {epoch}] Training Loss: {avg_loss:.4f}")
        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        psnr_list, ssim_list = [], []

        with torch.no_grad():
            for noisy_images, gt_images in val_loader:
                noisy_images = noisy_images.to(device)
                gt_images = gt_images.to(device)
                t = torch.randint(0, diffuser.T, (noisy_images.size(0),), device=device)

                pred_residual = model(noisy_images, t)
                denoised = (noisy_images - pred_residual).clamp(0, 1)

                metrics = evaluate_metrics(denoised, gt_images)
                psnr_list.append(metrics['PSNR'])
                ssim_list.append(metrics['SSIM'])

            avg_psnr = sum(psnr_list) / len(psnr_list)
            avg_ssim = sum(ssim_list) / len(ssim_list)

            print(f"[Epoch {epoch}] PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")
            print('----------------------------------------------')

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model (PSNR: {avg_psnr:.2f})")

            visualize_outputs(noisy_images, denoised, gt_images, epoch)


if __name__ == "__main__":
    print(device)
    # Paths
    train_cloud_dir = "../data/RICE/RICE1/cloud"
    train_label_dir = "../data/RICE/RICE1/label"

    test_cloud_dir = "../data/RICE/RICE1/Test/cloud"
    test_label_dir = "../data/RICE/RICE1/Test/label"


    print("Loading paths...")

    train_dataset = RICECloudDataset(train_cloud_dir, train_label_dir, transform=image_transforms)
    val_dataset = RICECloudDataset(test_cloud_dir, test_label_dir, transform=image_transforms)

    print("Number of training samples:", len(train_dataset))
    print("Number of validation samples:", len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)


    model = CloudDenoiser().to(device)
    diffuser = DiffusionProcess(T=1000)
    train(model, diffuser, train_loader, val_loader, num_epochs=50)
    # model.eval()
    # with torch.no_grad():
    #     sample_input = torch.randn(1, 3, 128, 128).to(device)  # Change size if needed
    #     sample_t = torch.randint(0, 1000, (1,), device=device)
    #     output = model(sample_input, sample_t)
    #     print(f"Sample input shape: {sample_input.shape}")
    #     print(f"Output shape: {output.shape}")  # Expect [1, 3, 128, 128]
    