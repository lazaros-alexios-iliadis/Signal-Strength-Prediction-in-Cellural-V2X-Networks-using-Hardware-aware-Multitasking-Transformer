import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from Transformer_model import MultiTaskFTTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from utils import mixup_data, relative_rmse, count_parameters, compute_flops
import matplotlib.pyplot as plt
import scipy.stats as stats
from hybrid_pruning import apply_hybrid_pruning, remove_weight_norm_all, PruningEnv
import seaborn as sns
from pruning_methods import compare_all_methods
from utils import measure_latency


# Load dataset
df_train = pd.read_csv("train_imputed.csv")

# Define features and targets
features = ['PCell_Downlink_Num_RBs', 'PCell_Downlink_Average_MCS',
            'SCell_Downlink_Num_RBs', 'SCell_Downlink_Average_MCS',
            'Latitude', 'Longitude', 'Altitude', 'speed_kmh', 'COG',
            'Traffic Jam Factor']

targets = ['PCell_RSRP_max', 'PCell_RSRQ_max', 'PCell_SNR_max',
           'SCell_RSRP_max', 'SCell_RSRQ_max', 'SCell_SNR_max']

# Extract feature matrix and target matrix
X = df_train[features].values
y = df_train[targets].values

# Validation-test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Normalize features using the training set statistics
original_y_mean = np.mean(y_test, axis=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)  # FIXED: Use transform, not fit_transform

target_scaler = StandardScaler()
y_train = target_scaler.fit_transform(y_train)
y_val = target_scaler.transform(y_val)
y_test = target_scaler.transform(y_test)

# Noise injection
noise_std = 0.05  # Standard deviation of noise
X_train += noise_std * np.random.randn(*X_train.shape)  # Add Gaussian noise
X_val += noise_std * np.random.randn(*X_val.shape)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create PyTorch DataLoaders
batch_size = 256
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
input_dim = len(features)

model = MultiTaskFTTransformer(
    input_dims=[input_dim] * 6,
    num_heads=8,
    num_layers=6,
    hidden_dim=128,
    output_dims=[1, 1, 1, 1, 1, 1]
).to(device)

# Compute model efficiency metrics
num_params = count_parameters(model)
flops = compute_flops(model)
# latency = measure_latency(model, device)
print(f"Total Trainable Parameters: {num_params:,}")
print(f"Total FLOPs: {flops:.2f} MFLOPs")
# print(f"Latency on {device}: {latency:.3f} ms")

# Define loss function & optimizer
criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# Training
num_epochs = 100
save_path = "best_ft_transformer.pth"
best_val_loss = float("inf")
patience = 10
counter = 0
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        batch_X_mix, batch_y_mix, lambda_mix = mixup_data(batch_X, batch_y, alpha=0.2)

        optimizer.zero_grad()
        batch_X_split = [batch_X_mix.clone() for _ in range(6)]
        predictions = model(batch_X_split)
        batch_y_mix = batch_y_mix.unsqueeze(1) if batch_y_mix.dim() == 1 else batch_y_mix
        loss = sum(criterion(pred, batch_y_mix[:, i].unsqueeze(1)) for i, pred in enumerate(predictions))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_X_split = [batch_X.clone() for _ in range(6)]
            predictions = model(batch_X_split)
            loss = sum(criterion(pred, batch_y[:, i].unsqueeze(1)) for i, pred in enumerate(predictions))
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(model.state_dict(), save_path)
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered!")
            break
    scheduler.step()

# Train - validation loss
plt.figure(figsize=(10, 6))  # Higher DPI for print quality

# Plot with different line styles for clarity
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker='o', linestyle='-', color='blue')
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker='s', linestyle='--', color='red')

# Publication-ready labels
plt.xlabel("Epochs", fontsize=14, fontweight='bold')
plt.ylabel("Loss", fontsize=14, fontweight='bold')
plt.title("Training & Validation Loss per Epoch", fontsize=16, fontweight='bold')

# Legend with improved placement
plt.legend(fontsize=12, loc="best")

# Grid formatting
plt.grid(True, linestyle='--', alpha=0.6)

# Border settings for a cleaner look
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig("training_validation_loss.pdf", bbox_inches="tight", transparent=True)

plt.show()

# Load best model
model.load_state_dict(torch.load(save_path))
model.to(device)

print("device:", device)
print("Model is on device:", next(model.parameters()).device)
if model is None:
    raise ValueError("Model loading failed! Check the saved model file.")
print("Model loaded:", model)

# Base latency
sample_batch = next(iter(test_loader))[0].to(device)
sample_input = [sample_batch.clone() for _ in range(6)]
latencies = [measure_latency(model, sample_input, device) for _ in range(10)]
baseline_latency = np.mean(latencies)
print(f"Avg Baseline Latency: {baseline_latency:.2f} ms")

model = remove_weight_norm_all(model)
if model is None:
    raise ValueError("Weight norm removal failed! Model is None.")
final_model = apply_hybrid_pruning(PruningEnv, model, test_loader, device, debug_mode=False)

final_model = final_model.cpu()  # Ensure quantized model is on CPU
device = torch.device("cpu")

# Evaluate
final_model.eval()
model.eval()
y_pred = []

with torch.no_grad():
    for batch_X, _ in test_loader:
        batch_X = batch_X.to(next(final_model.parameters()).device)
        batch_X_split = [batch_X.clone() for _ in range(6)]

        print(f"Input device: {batch_X.device}")
        print(f"Model device: {next(final_model.parameters()).device}")

        preds = final_model(batch_X_split)
        preds = torch.cat(preds, dim=1).cpu().numpy()
        y_pred.append(preds)


# Convert list of predictions into a NumPy array
y_pred = np.vstack(y_pred)  # Ensure correct shape

# Rescale Predictions to Original Scale**
y_pred_original = target_scaler.inverse_transform(y_pred)  # Convert back to original scale
y_test_original = target_scaler.inverse_transform(y_test)

print("Mean of y_test for each target:", np.mean(y_test_original, axis=0))

for i, target_name in enumerate(targets):
    mae = mean_absolute_error(y_test_original[:, i], y_pred_original[:, i])  # Compare rescaled predictions
    print(f" {target_name} MAE Loss: {mae:.4f}")
    # print(f" {target_name} MAE Loss: {mae[i]:.4f} (Relative to Mean: {mae[i] / mean_absolute_targets[i]:.2%})")

rrmse_values, avg_rrmse = relative_rmse(y_test_original, y_pred_original, original_y_mean)

# Print RRMSE for each output
for i, target_name in enumerate(targets):
    print(f" {target_name} RRMSE: {rrmse_values[i]:.4f}")

# Print Average RRMSE
print(f"\n Average RRMSE (ARRMSE): {avg_rrmse:.4f}")


# Residual and Q-Q plots
def plot_diagnostics(y_true, y_pred, task_name):
    residuals = y_true - y_pred

    # Residuals vs. Fitted
    plt.figure(figsize=(10, 5))
    plt.scatter(y_pred, residuals, alpha=0.6, color='blue')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Fitted Values", fontsize=14)
    plt.ylabel("Residuals", fontsize=14)
    plt.title(f"Residuals vs. Fitted: {task_name}", fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"residuals_vs_fitted_{task_name}.pdf", bbox_inches="tight", transparent=True)
    plt.show()

    # Q-Q Plot
    plt.figure(figsize=(10, 5))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot: {task_name}", fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"qq_plot_{task_name}.pdf", bbox_inches="tight", transparent=True)
    plt.show()


for i, task_name in enumerate(targets):
    plot_diagnostics(y_test_original[:, i], y_pred_original[:, i], task_name)

# Comparison
device = torch.device('cpu')
model = model.to(device)
final_model = final_model.to(device)

sample_input = [torch.randn(1, input_dim).to(device) for _ in range(6)]

results = compare_all_methods(
    base_model=model,
    val_loader=test_loader,
    train_loader=train_loader,
    device=device,
    sample_input=sample_input,
    hybrid_model=final_model
)


results_df = pd.DataFrame(results)

metrics = ['Loss', 'Latency(ms)']
normalized = results_df[metrics].copy()

for col in metrics:
    normalized[col] = (normalized[col] - normalized[col].min()) / (normalized[col].max() - normalized[col].min())

# Compute normalized score
normalized['Normalized Score'] = (
    0.5 * normalized['Loss'] +
    0.5 * normalized['Latency(ms)']
)

# Attach and sort
results_df['Normalized Score'] = normalized['Normalized Score']
results_df = results_df.sort_values(by='Normalized Score').reset_index(drop=True)

# Print nicely
print("\n Final Benchmark with Normalized Score:")
print(results_df.to_markdown(index=False))

# Radar needs values to loop back
# radar_metrics = ['Loss', 'Latency(ms)', 'FLOPs', 'Sparsity(%)']
radar_metrics = ['Loss', 'Latency(ms)']
angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
angles += angles[:1]

# Create radar plot
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

for i, row in normalized.iterrows():
    values = row[radar_metrics].tolist()   # Use only relevant metrics
    values += values[:1]                   # Close the loop
    ax.plot(angles, values, label=results_df['Method'][i], linewidth=2)
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_metrics)
ax.set_title("Normalized Performance")
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig("radar_plot.pdf")
plt.show()


ranked_df = results_df.copy()
# for col in ['Loss', 'Latency(ms)', 'FLOPs', 'Sparsity(%)']:
for col in ['Loss', 'Latency(ms)']:
    if col == 'Sparsity(%)':
        ranked_df[col] = ranked_df[col].rank(ascending=False)
    else:
        ranked_df[col] = ranked_df[col].rank(ascending=True)

ranked_df.set_index('Method', inplace=True)

plt.figure(figsize=(8, 5))
sns.heatmap(ranked_df, annot=True, cmap='YlGnBu', cbar=True, fmt=".3f")
plt.title("Relative Ranking of Pruning Methods")
plt.tight_layout()
plt.savefig("ranking_heatmap.pdf")
plt.show()

plt.figure(figsize=(8, 8))
sns.barplot(data=results_df, x='Normalized Score', y='Method', palette='viridis')
plt.title("Normalized Composite Score by Method")
plt.xlabel("Score")
plt.grid(True)
plt.tight_layout()
plt.savefig("normalized_score.pdf")
plt.show()
