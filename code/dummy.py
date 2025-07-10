import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from denoiser_SE import CloudDenoiser  # Update if filename is different

# Paths (SET THESE)
cloud_path = "../data/RICE/RICE1/Test/cloud/492.png"
label_path = "../data/RICE/RICE1/Test/label/492.png"
model_path = "seb_best_model.pth"
output_dir = "denoised_output"
os.makedirs(output_dir, exist_ok=True)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CloudDenoiser().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load image
cloud_img = transform(Image.open(cloud_path).convert("RGB")).unsqueeze(0).to(device)
true_img = transform(Image.open(label_path).convert("RGB")).unsqueeze(0).to(device)
timestep = torch.randint(0, 1000, (1,), device=device)

with torch.no_grad():
    pred_noise = model(cloud_img, timestep)
    denoised = (cloud_img - pred_noise).clamp(0, 1)

def to_img(tensor):
    return tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

plt.figure(figsize=(12, 4))
titles = ["Cloudy", "Denoised", "Label"]
images = [cloud_img, denoised, true_img]

for i, (img, title) in enumerate(zip(images, titles)):
    plt.subplot(1, 3, i + 1)
    plt.imshow(to_img(img))
    plt.title(title)
    plt.axis("off")

plt.tight_layout()

# Save the figure using the cloud image filename
filename = os.path.basename(cloud_path).replace(".png", "_comparison.png")
save_path = os.path.join(output_dir, filename)
plt.savefig(save_path)
plt.show()

print(f"Saved comparison to {save_path}")
