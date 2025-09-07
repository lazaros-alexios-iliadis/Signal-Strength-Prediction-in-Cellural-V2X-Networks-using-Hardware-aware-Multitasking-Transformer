import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from hybrid_pruning import apply_hybrid_pruning
from utils import compute_flops, measure_latency
import torch.ao.quantization as quant


class QuantWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.feature_tokenizers = model.feature_tokenizers
        self.encoder = model.transformer_encoder
        self.heads = model.output_heads

    def forward(self, x_list):
        outputs = []
        for i, x in enumerate(x_list):
            x_tok = self.feature_tokenizers[i](x)
            x_enc = self.encoder(x_tok)
            x_pool = x_enc.mean(dim=1)
            out = self.heads[i](x_pool)
            outputs.append(out)
        return outputs


def evaluate_model(model, val_loader, device):
    model.eval()
    model.to(device)

    preds_all = [[] for _ in range(6)]
    targets_all = [[] for _ in range(6)]

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_X_list = [batch_X.clone() for _ in range(len(model.feature_tokenizers))]
            preds = model(batch_X_list)

            for i in range(6):
                preds_all[i].append(preds[i].cpu())
                targets_all[i].append(batch_y[:, i:i+1].cpu())

    # Stack predictions and targets per output
    rrmse_values = []
    for i in range(6):
        pred = torch.cat(preds_all[i])
        target = torch.cat(targets_all[i])
        rmse = torch.sqrt(torch.mean((pred - target) ** 2))
        mean_target = torch.mean(torch.abs(target)) + 1e-8  # avoid division by 0
        rrmse = rmse / mean_target
        rrmse_values.append(rrmse.item())

    avg_rrmse = sum(rrmse_values) / len(rrmse_values)

    return avg_rrmse


def apply_lottery_ticket(model, device, prune_amount=0.3):
    model = deepcopy(model).to(device)  # Ensure model is on correct device
    for module in model.modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=prune_amount)
            prune.remove(module, 'weight')
    return model


def apply_movement_pruning(model, train_loader, device, prune_amount=0.3):
    model = deepcopy(model).to(device)
    model.train()  # Enable training mode
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_X, batch_y = next(iter(train_loader))
    batch_X = batch_X.to(device)
    batch_y = batch_y.to(device)

    batch_X_list = [batch_X.clone() for _ in range(len(model.feature_tokenizers))]

    for param in model.parameters():
        param.requires_grad = True

    outputs = model(batch_X_list)
    loss_fn = nn.SmoothL1Loss()
    loss = sum(loss_fn(p, batch_y[:, i:i+1]) for i, p in enumerate(outputs))

    # Make sure no no_grad context is active
    loss.backward()

    for module in model.modules():
        if isinstance(module, nn.Linear) and module.weight.grad is not None:
            movement_score = (module.weight.data * module.weight.grad).abs()
            threshold = torch.quantile(movement_score.view(-1), prune_amount)
            mask = (movement_score >= threshold).float()
            module.weight.data *= mask

    return model


def apply_mixed_precision_quant(model, sample_input):
    print("[INFO] Applying mixed-precision quantization...")

    model = deepcopy(model).cpu().eval()

    # Quantize only feature tokenizers and output heads
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.qconfig = quant.default_dynamic_qconfig

    # Apply quantization to the feature tokenizers and heads
    model.feature_tokenizers = nn.ModuleList([
        quant.quantize_dynamic(ftok, {nn.Linear}, dtype=torch.qint8)
        for ftok in model.feature_tokenizers
    ])
    model.output_heads = nn.ModuleList([
        quant.quantize_dynamic(head, {nn.Linear}, dtype=torch.qint8)
        for head in model.output_heads
    ])

    # Wrap the quantized model to preserve forward behavior
    quantized_model = QuantWrapper(model)
    quantized_model.eval()

    print("[INFO] Quantized model created.")
    return quantized_model


def apply_layerdrop(model, drop_prob=0.3):
    model = deepcopy(model)
    layers = list(model.transformer_encoder.layers)
    kept_layers = [layer for layer in layers if torch.rand(1).item() > drop_prob]

    # Fallback: Keep at least one layer
    if not kept_layers:
        kept_layers = [layers[0]]

    model.transformer_encoder.layers = nn.ModuleList(kept_layers)
    print(f" LayerDrop kept {len(kept_layers)} out of {len(layers)} layers.")
    return model


def benchmark_model(name, model, val_loader, device, sample_input):
    # Special handling for quantized models
    if "Quant" in name:
        device = 'cpu'
        model = model.cpu()

        # Convert sample input to CPU tensors
        sample_input = [x.cpu() if isinstance(x, torch.Tensor) else torch.tensor(x).cpu()
                        for x in sample_input]

        # Convert validation data to CPU
        cpu_X, cpu_y = [], []
        for x, y in val_loader:
            cpu_X.append(x.cpu())
            cpu_y.append(y.cpu())

        cpu_X = torch.cat(cpu_X, dim=0)
        cpu_y = torch.cat(cpu_y, dim=0)
        cpu_val_loader = DataLoader(
            TensorDataset(cpu_X, cpu_y),
            batch_size=val_loader.batch_size,
            shuffle=False
        )
        val_loader = cpu_val_loader

    model.eval()
    model = model.to(device)

    with torch.no_grad():
        loss = evaluate_model(model, val_loader, device)

    flops = compute_flops(model)
    latency = measure_latency(model, sample_input, device)

    total_params = sum(p.numel() for p in model.parameters())
    zero_params = sum((p == 0).sum().item() for p in model.parameters())
    sparsity = 100 * zero_params / total_params

    return {
        'Method': name,
        'Loss': round(loss, 4),
        'FLOPs': round(flops, 2),
        'Latency(ms)': round(latency, 2),
        'Sparsity(%)': round(sparsity, 2)
    }


def compare_all_methods(base_model, val_loader, train_loader, device, sample_input, hybrid_model=None):

    results = []

    # 1. Lottery Ticket
    print(" Running LTH...")
    lth_model = apply_lottery_ticket(deepcopy(base_model), prune_amount=0.3, device=device)
    results.append(benchmark_model("Lottery Ticket", lth_model, val_loader, device, sample_input))

    # 2. Movement Pruning
    print(" Running Movement Pruning...")
    move_model = apply_movement_pruning(deepcopy(base_model), train_loader, device, prune_amount=0.3)
    results.append(benchmark_model("Movement Pruning", move_model, val_loader, device, sample_input))

    # 3. Mixed-Precision Quantization
    print(" Running Quantization...")
    quant_model = apply_mixed_precision_quant(deepcopy(base_model), sample_input)
    results.append(benchmark_model("Mixed-Precision Quantization", quant_model, val_loader, device, sample_input))

    print(" Running LayerDrop...")
    ld_model = apply_layerdrop(deepcopy(base_model), drop_prob=0.3)
    results.append(benchmark_model("LayerDrop", ld_model, val_loader, device, sample_input))

    print(" Running Hybrid Pruning (Ours)...")
    if hybrid_model is None:
        hybrid_model = apply_hybrid_pruning(deepcopy(base_model), val_loader, device=device)

    results.append(benchmark_model("Hybrid Pruning (Ours)", hybrid_model, val_loader, device, sample_input))

    # Create CPU version of validation loader
    X_cpu, y_cpu = [], []
    for x, y in val_loader:
        X_cpu.append(x.cpu())
        y_cpu.append(y.cpu())

    X_cpu = torch.cat(X_cpu, dim=0)
    y_cpu = torch.cat(y_cpu, dim=0)
    cpu_dataset = TensorDataset(X_cpu, y_cpu)
    cpu_val_loader = DataLoader(cpu_dataset, batch_size=val_loader.batch_size, shuffle=False)
    cpu_sample_input = [x.cpu() for x in sample_input]


    return results
