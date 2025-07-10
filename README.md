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

## Architecture Overview

### Stage 1: SE-DeCloud

- U-Net with Squeeze-and-Excitation (SE) blocks
- Sinusoidal timestep embeddings (inspired by DDPMs)
- Trained on RICE1 → Fine-tuned on EuroSAT with synthetic clouds (Perlin noise)

### Stage 2: ConvEA-ViT

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

---


## Contributors

- Prithviraajan Senthilkumar  
- Hrishikesh Virupakshi

---

## License

- This project is licensed under the **MIT License**.
