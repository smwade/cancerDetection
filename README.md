# Cervical Cancer Detection

Automated cervical cancer detection from Pap smear slides using deep learning. This project tackles the challenge of making cancer screening accessible to low-resource communities by building models that can classify cell images as normal, abnormal, or benign.

## Motivation

Cervical cancer is one of the deadliest cancers among women worldwide, yet it is also one of the most preventable. Using Pap smear tests, trained technicians can detect cancerous cells early and provide life-saving treatment. In 2018 alone, there were over 570,000 cases of cervical cancer, with a staggering 90% occurring in low- to middle-income countries. This highlights a critical need: cost-effective, automated screening.

## Approach

This project explores multiple deep learning architectures for cell classification:

- **CNN** - Baseline convolutional neural network
- **U-Net** - Segmentation-based approach
- **Pix2Pix** - GAN-based synthetic data generation
- **SinGAN** - Single image GAN for data augmentation

A key challenge is the lack of labeled data. Much of this work focuses on augmenting existing datasets and creating synthetic training data using the [MediAug](https://github.com/smwade/MediAug) toolkit.

## Project Structure

```
cancerDetection/
  models/
    cnn/          # Baseline CNN classifier
    pix2pix/      # GAN for mask-to-image generation
    singan/       # Single image GAN augmentation
    unet/         # U-Net segmentation model
  notebooks/      # Data exploration and visualization
  scripts/        # Training and evaluation scripts
  MediAug/        # Data augmentation submodule
```

## Datasets

### SMEAR

917 individual cells segmented by nucleus and cytoplasm.
[Download](https://mde-lab.aegean.gr/downloads)

### SIPaKMeD

4,049 images of isolated cells manually cropped from 966 cluster cell images, divided into five categories (normal, abnormal, benign).
[Download](http://cs.uoi.gr/~marina/sipakmed.html)

## Installation

```bash
git clone https://github.com/smwade/cancerDetection
cd cancerDetection
pip install -r requirements.txt
```

### Data Preparation

```bash
python prepare_data.py \
  --input_dir data/SIPaKMeD/ \
  --out_dir data/sipakmed_processed
```

## Related Work

- [MediAug](https://github.com/smwade/MediAug) - The data augmentation toolkit built alongside this project
