import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import time


def mixup_data(x, y, alpha=0.2):

    if alpha > 0:
        lambda_mix = np.random.beta(alpha, alpha)  # Sample lambda from Beta distribution
    else:
        lambda_mix = 1  # No mixup applied

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)  # Randomly shuffle batch indices

    # Apply Mixup
    x_mix = lambda_mix * x + (1 - lambda_mix) * x[index, :]
    y_mix = lambda_mix * y + (1 - lambda_mix) * y[index, :]

    return x_mix, y_mix, lambda_mix


def relative_rmse(y_true, y_pred, original_y_mean):

    # Compute RMSE for each target
    mse = np.mean((y_true - y_pred) ** 2, axis=0)  # MSE per target
    rmse = np.sqrt(mse)  # RMSE per target

    # Use the original mean (before scaling) to compute RRMSE
    mean_target = np.abs(original_y_mean)  # Use absolute mean of original targets

    # Avoid division by zero (replace near-zero means with small value)
    mean_target[mean_target < 1e-8] = 1e-8  # Prevent extreme values

    # Compute RRMSE per target
    rrmse_values = rmse / mean_target

    # Compute ARRMSE
    avg_rrmse = np.mean(rrmse_values)

    return rrmse_values, avg_rrmse


def prune_model(model, pruning_amount=0.3):

    # Prune Linear layers in the Transformer Encoder
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):  # Prune fully connected layers
            prune.l1_unstructured(module, name="weight", amount=pruning_amount)
            prune.remove(module, "weight")  # Permanently remove pruned weights

    return model


def relative_rmse_distinct(y_true, y_pred, original_y_mean, epsilon=1e-8):

    # Compute RMSE for each target
    mse = np.mean((y_true - y_pred) ** 2, axis=0)  # Compute MSE per target
    rmse = np.sqrt(mse)  # Compute RMSE per target

    mean_target = np.abs(original_y_mean)  # Use absolute mean to avoid negative scaling
    mean_target[mean_target < epsilon] = epsilon  # Prevent division by zero

    rrmse_values = rmse / mean_target  # Compute RRMSE per target
    avg_rrmse = np.mean(rrmse_values)  # Compute average RRMSE

    return rrmse_values, avg_rrmse


def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_flops(model):
    total_flops = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.detach()
            non_zero = (weight != 0).float().sum().item()
            # Each weight contributes 2 FLOPs (1 multiply + 1 add)
            total_flops += 2 * non_zero
    return total_flops / 1e6


def measure_latency(model, sample_input, device, warmup=10, runs=50):
    model = model.to(device).eval()
    sample_input = [x.to(device) for x in sample_input]

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample_input)

    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(sample_input)
    end = time.time()

    return ((end - start) / runs) * 1000  # ms


def move_to_model_device(model, batch):
    """Moves a batch to the same device as the model."""
    model_device = next(model.parameters()).device
    return batch.to(model_device)


def compute_custom_score(loss, latency, weights=None):
    """
    Combine objectives into a single score for ranking.
    Lower score = better.
    """

    if weights is None:
        weights = {
            "loss": 0.5,
            "latency": 0.5
        }

    # Normalize each term (avoid division by zero)
    loss_norm = loss / (loss + 1e-8)
    latency_norm = latency / (latency + 1e-8)

    # Compute weighted sum
    score = (
        weights["loss"] * loss_norm +
        weights["latency"] * latency_norm
    )

    return score

