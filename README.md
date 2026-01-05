# PyTorch Playground

A hands-on playground for **learning and experimenting with machine learning using PyTorch**.

This repository will contain implementations of a wide range of machine learning models — from simple **MLPs** to **Computer Vision models**, **NLP models**, **Generative models**, and more — all built with for learning purposes.

The goal is to **understand how things work under the hood** by:
1. Implementing models **from scratch**
2. Gradually refactoring them to use **PyTorch abstractions**
3. Optimizing and benchmarking them on **well-known datasets**

---

## Motivation

Modern ML frameworks hide a lot of complexity. While this is great for productivity, it can make learning harder.

This repo is meant to bridge that gap by:
- Re-implementing core ideas manually (forward pass, backprop intuition, training loops)
- Transitioning to idiomatic PyTorch (`nn.Module`, `DataLoader`, etc.)
- Comparing performance, readability, and accuracy across implementations

Everything here is **educational**, exploratory, and experiment-driven.

---

## What’s Inside

The repository will cover **all major ML domains**, including but not limited to:

### Core Models
- Multi-Layer Perceptrons (MLPs)
- Custom training loops & optimizers

### Computer Vision
- CNNs from scratch
- Modern architectures (ResNet, Vision Transformers, etc.)
- Training on datasets like MNIST, CIFAR-10/100, ImageNet (where feasible)

### Natural Language Processing
- RNNs, LSTMs, GRUs
- Transformers (from scratch → `nn.Transformer`)
- Text classification, language modeling, sequence-to-sequence tasks

### Generative Models
- Autoencoders & VAEs
- GANs
- Image and text generation experiments

### Training & Optimization
- Manual vs PyTorch optimizers
- Regularization techniques
- Hyperparameter tuning
- Accuracy and loss tracking

---

## Learning Philosophy

Each model typically follows this progression:

1. **From scratch**
   - Minimal PyTorch usage
   - Manual forward logic
   - Explicit training loops
2. **PyTorch-native**
   - `nn.Module`
   - Built-in layers & losses
   - Clean, reusable code
3. **Optimization & benchmarking**
   - Improved accuracy
   - Better training stability
   - Comparison with known baselines

This makes it easy to see *why* PyTorch abstractions exist and *how* they help.

---

## Repository Structure (example)

```text
pytorch-playground/
├── MLP_Manual/
│   ├── dataset/
│   ├── utils/
│   ├── README.md (Info About Implementation)
│   ├── train.py
│   └── test.py
├── CNN_Manual/
│   ├── dataset/
│   ├── utils/
│   ├── README.md (Info About Implementation)
│   ├── train.py
│   └── test.py
...   
│
└── README.md
```

---

## Datasets

Common datasets used throughout the repository include:

- **MNIST** – handwritten digit classification  
- **CIFAR-10 / CIFAR-100** – small-scale image classification  
- **Fashion-MNIST** – clothing image classification  
- **IMDB** – sentiment analysis  
- **Custom datasets** – for controlled experiments and debugging

Datasets are typically downloaded automatically or stored locally under the `datasets/` directory inside each model implementation directory. Each experiment documents the dataset it uses.

---

## Tech Stack

- **Python**
- **PyTorch**
- torchvision / torchtext
- NumPy
- Pandas
- Kornia
- Matplotlib (for visualization)

---

## Disclaimer

This repository is **not production-focused**.

- Code prioritizes **clarity over performance**
- Some implementations are intentionally verbose for learning purposes
- Models are not shared, but the accuracy achieved will be documented inside README.md file inside each model implementation directory

---

## Summary

This repository is a hands-on playground designed to help me build a better intuition for Machine Learning / Deep Learning by implementing famous architectures and models.

---

Implementation by **Denis Gracijan Bosika**
