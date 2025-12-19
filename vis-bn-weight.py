import matplotlib.pyplot as plt
import numpy as np
import torch

from ultralytics import YOLO


def extract_bn_gamma(model):
    """Extract BN gamma weights"""
    gamma_weights = []
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            gamma_weights.append(m.weight.detach().cpu().numpy())
    return np.concatenate(gamma_weights)


def visualize_bn_gamma_distribution(model):
    """
    Extract gamma weights from Batch Normalization layers of the model
    and visualize their distribution.

    Args:
        model: The deep learning model (e.g., PyTorch or TensorFlow/Keras model).
    """
    gamma_weights = []

    # Example for PyTorch model
    for layer in model.modules():
        if hasattr(layer, 'weight') and isinstance(layer, torch.nn.BatchNorm2d):
            gamma_weights.append(layer.weight.detach().cpu().numpy())

    # Flatten the list of gamma weights
    gamma_weights = np.concatenate(gamma_weights)

    # Plot the distribution
    plt.figure(figsize=(8, 5))
    plt.hist(gamma_weights, bins=100, color='blue', alpha=0.7)
    plt.title("Distribution of Gamma Weights (BN Layers)")
    plt.xlabel("Gamma Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    # plt.savefig('bn-weight-distribution-original.jpg')
    plt.savefig('bn-weight-distribution-sparsity.jpg')

def plot_bn_gamma_hist(gamma_normal_vals, gamma_sparse_vals, save_path):
    plt.figure(figsize=(8, 5))

    plt.hist(
        gamma_normal_vals,
        bins=200,
        density=True,
        alpha=0.6,
        color="black",
        label="Normal Training"
    )

    plt.hist(
        gamma_sparse_vals,
        bins=200,
        density=True,
        alpha=0.6,
	color="crimson",
        label="Sparsity Training"
    )

    plt.xlim(0, 6)
    plt.xlabel("BN Gamma")
    plt.ylabel("Density")
    plt.title("BN Gamma Distribution Shift")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

def plot_hist_and_heatmap(gamma_normal_vals, gamma_sparse_vals, save_path):
    bins = 200
    x_min, x_max = -6, 6

    # Compute density histograms
    hist_normal, _ = np.histogram(
        gamma_normal_vals, bins=bins, range=(x_min, x_max), density=True
    )
    hist_sparse, _ = np.histogram(
        gamma_sparse_vals, bins=bins, range=(x_min, x_max), density=True
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 3), sharey=True)

    # Heatmap - Normal
    im1 = axes[0].imshow(
        hist_normal[np.newaxis, :],
        aspect="auto",
        cmap="hot",
        extent=[x_min, x_max, 0, 1]
    )
    axes[0].set_title("Normal Training")
    axes[0].set_yticks([])
    axes[0].set_xlabel("BN Gamma")

    # Heatmap - Sparsity
    im2 = axes[1].imshow(
        hist_sparse[np.newaxis, :],
        aspect="auto",
        cmap="hot",
        extent=[x_min, x_max, 0, 1]
    )
    axes[1].set_title("Sparsity Training")
    axes[1].set_yticks([])
    axes[1].set_xlabel("BN Gamma")

    # Shared colorbar
    fig.colorbar(im2, ax=axes, label="Density")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


if __name__ == '__main__':
    # weight = "weights/orignal.pt"
    # weight = "runs/train-sparsity2/weights/last.pt"
    # model = YOLO(weight)
    # visualize_bn_gamma_distribution(model)
    weight_normal = "weights/original.pt"
    weight_sparse = "runs/train-sparsity/weights/last.pt"

    model_normal = YOLO(weight_normal).model
    model_sparse = YOLO(weight_sparse).model

    gamma_normal = extract_bn_gamma(model_normal)
    gamma_sparse = extract_bn_gamma(model_sparse)

    plot_bn_gamma_hist(
        gamma_normal,
        gamma_sparse,
        save_path="assets/bn_gamma_hist.jpg"
    )

    plot_hist_and_heatmap(
        gamma_normal,
        gamma_sparse,
        save_path="assets/bn_gamma_hist_heatmap.jpg"
    )