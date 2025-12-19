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

def visualize_two_models(weight_normal, weight_sparse, save_path):
    model_normal = YOLO(weight_normal).model
    model_sparse = YOLO(weight_sparse).model

    gamma_normal = extract_bn_gamma(model_normal)
    gamma_sparse = extract_bn_gamma(model_sparse)

    plt.figure(figsize=(8, 5))

    plt.hist(
        gamma_normal,
        bins=200,
        density=True,
        alpha=0.6,
        color="black",      
        label="Normal Training"
    )

    plt.hist(
        gamma_sparse,
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

if __name__ == '__main__':
    # weight = "weights/orignal.pt"
    # weight = "runs/train-sparsity2/weights/last.pt"
    # model = YOLO(weight)
    # visualize_bn_gamma_distribution(model)

    visualize_two_models(
        weight_normal="weights/original.pt",
        weight_sparse="runs/train-sparsity/weights/last.pt",
        save_path="assets/bn_gamma_comparison.jpg"
    )