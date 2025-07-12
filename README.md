# Satellite Image Denoising & Classification using SE-DeCloud and ConvEA-ViT

>  **Accepted at CVIP 2025 – Conference on Computer Vision and Image Processing**

This project presents a hybrid two-stage deep learning framework that performs **cloud removal** and **land cover classification** on satellite imagery. The method combines a **diffusion-inspired U-Net denoiser** with an **edge-aware Vision Transformer**, achieving state-of-the-art results on RICE1 and EuroSAT datasets.

---

## Key Highlights

- **SE-DeCloud**: Diffusion-style cloud removal model based on U-Net + Squeeze-and-Excitation (SE) blocks.
- **ConvEA-ViT**: Convolution-enhanced Vision Transformer that integrates edge-aware self-attention.
- Trained on:
  - **RICE1** (cloud removal)
  - **EuroSAT** (classification with synthetic cloud noise using Perlin noise)
- Accepted at **CVIP 2025**


## Dataset & Preprocessing
Place the downloaded data into the corresponding folders.
- **RICE1**: Paired cloudy & cloud-free satellite images  
  📎 [https://github.com/likyoo/RICE1](https://github.com/likyoo/RICE1)

- **EuroSAT**: 27,000 RGB satellite images, 10 land use classes  
  📎 [https://github.com/phelber/EuroSAT](https://github.com/phelber/EuroSAT)

- **Cloud Simulation**: Perlin noise added to simulate atmospheric occlusion on EuroSAT.

---

## ⚙️ Setup Instructions

To install all dependencies required for this project, use the provided `requirements.txt` file.

### 🔧 Installation with pip

```bash
pip install -r requirements.txt

```

---

## Architecture Overview

![image](https://github.com/user-attachments/assets/c99bf8ad-4ba6-4567-9937-5a708650b497)

### Stage 1: SE-DeCloud

![image](https://github.com/user-attachments/assets/21f316de-bc14-461d-b657-3fbb84921233)

- U-Net with Squeeze-and-Excitation (SE) blocks
- Sinusoidal timestep embeddings (inspired by DDPMs)
- Trained on RICE1 → Fine-tuned on EuroSAT with synthetic clouds (Perlin noise)

### Stage 2: ConvEA-ViT

![image](https://github.com/user-attachments/assets/bed9e971-91f5-4e8e-bba0-fadda0ec8a9b)

- Sobel edge maps are fused via convolution
- Vision Transformer attends to both edge-aware and global semantic features
- Enhanced classification performance on occluded or low-quality inputs

---


## Performance Metrics

| Task               | Metric       | Score     |
|--------------------|--------------|-----------|
| Cloud Removal      | PSNR         | 29.87 dB  |
|                    | SSIM         | 0.9641    |
| Classification     | Accuracy     | 98.14%    |
|                    | F1 Score     | 97.8%     |

ConvEA-ViT outperforms baseline CNNs (ResNet, DenseNet) and matches performance of advanced transformers like ViT and SwinT, with fewer parameters.

## Results

![image](https://github.com/user-attachments/assets/6fd28bc8-9de7-441e-9c73-2b83a4ba0441)

Each triplet includes (top to bottom): the original cloudy image, the corresponding denoised image, and the ground truth reference.

---

## License
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
- This project is licensed under the **MIT License**.
- You are free to use, modify, and distribute the code with proper attribution.
- See the [`LICENSE`](LICENSE) file for full details.
