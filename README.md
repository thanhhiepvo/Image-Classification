# Image Classification: From Baseline CNNs to Vision Transformers

This repository features a comprehensive suite of **Image Classification** models implemented in **PyTorch**, spanning across fundamental baselines, custom architectures, and State-of-the-Art (SOTA) Vision Transformers.


---

## 🧠 Project Scope & Methodology

This project covers the full spectrum of image classification techniques, emphasizing **from-scratch architectural implementations** over high-level library calls.

### 🧩 1. Manual Architecture Implementations
Built from first principles using standard PyTorch `nn.Module` layers:
-   **ResNet-10**: Custom residual blocks with skip connections and projection layers for dimension alignment (e.g., in `ResNet_8_CIFAR10.ipynb`).
-   **Inception Blocks**: Implementation of V1.5 and V2 multi-branch inception modules with 1x1, 3x3, and 5x5 filters (e.g., in `Inception_CIFAR10.ipynb`).
-   **Vision Transformer (ViT)**: A full implementation of the ViT architecture, including:
    *   **Patch Embedding** (splitting images into 1D sequences).
    *   **Multi-head Self-Attention (MSA)**.
    *   **Transformer Encoder Blocks** (Attention + MLP).
    *   **CLS Token & Position Embeddings**.

### 🚀 2. Advanced Methods & Transfer Learning
-   **EfficientNet Fine-tuning**: Applying **EfficientNet-B0** pre-trained on ImageNet to a custom 3-class dataset (Pizza, Steak, Sushi) using both feature extraction and fine-tuning strategies.
-   **Pre-trained ViT**: Comparing custom-built Transformer performance against official pre-trained ViT-Base models.

### 📊 3. Baselines & Datasets
-   **MNIST & Fashion-MNIST**: Simple CNN baselines for initial experimentation and optimization.
-   **CIFAR-10**: Testing complex architectures (ResNet, Inception, SqueezeNet) on color imagery benchmarks.

---

## 📊 Numerical Performance Benchmarks

This project achieves competitive accuracy across various datasets, demonstrating the effectiveness of custom-built architectures compared to standard baselines.

| Dataset | Architecture | Epochs | Accuracy | Key Technical Note |
| :--- | :--- | :--- | :--- | :--- |
| **CIFAR-10** | Custom ResNet-10 | 10 | **~74.3%** | Residual layers successfully resolved vanishing gradients. |
| **CIFAR-10** | SqueezeNet (Manual) | 5 | **~69.0%** | Achieved high recall with 50x fewer parameters via Fire Blocks. |
| **Fashion-MNIST**| TinyVGG-Style CNN | 3 | **~88.4%** | Fast convergence on grayscale patterns using BatchNorm. |
| **Food (3-Class)**| EfficientNet-B0 | 5 | **90%+** | Demonstrated high-impact Transfer Learning on small data. |
| **Custom Data** | ViT (From Scratch) | 10 | Competitive | Manual MSA & Patch Embedding implementation. |

---

## 🎯 Impact & Technical Contributions

- **Architectural Mastery**: Engineered complex models like **ResNet**, **Inception**, and **Vision Transformers (ViT)** from first principles. This proves a deep understanding of multi-branch networks, residual mappings, and self-attention mechanisms.
- **Optimization Expertise**: Implemented advanced training pipelines including **Weight Decay**, **Dropout**, **Learning Rate Scheduling (OneCycleLR)**, and **Label Smoothing** to improve generalization.
- **Versatile Problem Solving**: Showcased the ability to handle various data types (MNIST, CIFAR, custom high-res images) and model requirements (from lightweight SqueezeNet to computationally heavy ViTs).

---

## 🛠️ Tools & Dependencies
- **Python 3.11+**
- **PyTorch & TorchVision**
- **TorchInfo** (model visualization)
- **Matplotlib/Seaborn** (performance analytics)

## 🚀 How to Run
1. Open any of the `.ipynb` notebooks in VS Code or Google Colab.
2. GPU acceleration is highly recommended for **ViT** and **ResNet** notebooks.
3. Each notebook includes its own data-loading and preprocessing pipeline for immediate execution.
