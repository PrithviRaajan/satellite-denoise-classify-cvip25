import os
import glob
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader, random_split, Dataset
from perlin_noise import PerlinNoise
import matplotlib.pyplot as plt
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from PIL import Image
from torchmetrics.functional import structural_similarity_index_measure as ssim

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
        self.noise = PerlinNoise(octaves=6, seed=42)

    def add_cloud_noise(self, images, t):
        B, C, H, W = images.shape
        perlin_noise = torch.zeros((B, 1, H, W), device=images.device)
        for b in range(B):
            for i in range(H):
                for j in range(W):
                    perlin_noise[b, 0, i, j] = self.noise([i / H, j / W])
        perlin_noise = perlin_noise.expand(-1, C, -1, -1)  # Match channels

        alpha = t.float().view(-1, 1, 1, 1) / self.T
        noisy_images = (1 - alpha) * images + alpha * perlin_noise
        return noisy_images, perlin_noise


def evaluate_metrics(pred, target):
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    return {
        'PSNR': psnr(pred, target, data_range=1.0).item(),
        'SSIM': ssim(pred, target, data_range=1.0).item()
    }

def visualize_outputs(noisy, pred, target, epoch, save_dir='visuals_seb'):
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

def train(model, diffuser, train_loader, val_loader, num_epochs=25, save_path='finetuned_eurosat.pth'):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_psnr = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        for images, _ in train_loader:
            images = images.to(device)
            t = torch.randint(0, diffuser.T, (images.size(0),), device=device)
            noisy_images, _ = diffuser.add_cloud_noise(images, t)
            pred_clean = model(noisy_images, t)
            loss = nn.MSELoss()(pred_clean, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch}] Loss: {total_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        psnr_list, ssim_list = [], []
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                t = torch.randint(0, diffuser.T, (images.size(0),), device=device)
                noisy_images, _ = diffuser.add_cloud_noise(images, t)
                denoised = model(noisy_images, t).clamp(0, 1)
                metrics = evaluate_metrics(denoised, images)
                psnr_list.append(metrics['PSNR'])
                ssim_list.append(metrics['SSIM'])

            avg_psnr = sum(psnr_list) / len(psnr_list)
            avg_ssim = sum(ssim_list) / len(ssim_list)
            print(f"[Epoch {epoch}] PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                torch.save(model.state_dict(), save_path)
                print(f">>> Saved best model at epoch {epoch} with PSNR: {avg_psnr:.2f}")

            if epoch % 5 == 0:
                visualize_outputs(noisy_images, denoised, images, epoch)


if __name__ == "__main__":
    model = CloudDenoiser().to(device)
    model.load_state_dict(torch.load("seb_best_model.pth", map_location=device))
    print("Loaded pretrained model.")
    
    data_root = "../data/EuroSAT"
    dataset = datasets.ImageFolder(root=data_root, transform=image_transforms)

    print("Loading and splitting data.")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    print("Model Training.")
    diffuser = DiffusionProcess(T=1000)
    train(model, diffuser, train_loader, val_loader, num_epochs=20)
    finetuned_model_path = 'finetuned_model.pth'
    torch.save(model.state_dict(), finetuned_model_path)
    print(f"Finetuned model saved to {finetuned_model_path}")