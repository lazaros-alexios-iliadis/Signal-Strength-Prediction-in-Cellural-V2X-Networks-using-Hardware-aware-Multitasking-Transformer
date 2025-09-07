import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from copy import deepcopy
from stable_baselines3 import PPO
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from utils import measure_latency
import gymnasium as gym
from gymnasium import spaces
import torch.nn.utils.parametrize as parametrize
import matplotlib.pyplot as plt
from custom_policy import CustomMLPExtractor
from torch.quantization import quantize_dynamic
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from pymoo.core.population import Population
from pymoo.operators.sampling.rnd import FloatRandomSampling


# Utility: Evaluate Model Performance
def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_model(model, val_loader, device):
    """Evaluates the model and returns validation loss."""
    model.eval()
    total_loss = 0
    criterion = nn.SmoothL1Loss()

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_X_split = [batch_X.clone() for _ in range(6)]
            preds = model(batch_X_split)
            loss = sum(criterion(pred, batch_y[:, i].unsqueeze(1)) for i, pred in enumerate(preds))
            total_loss += loss.item()

    return total_loss / len(val_loader)


def apply_safe_dynamic_quantization(model):
    """
    Quantizes only FeatureTokenizer and OutputHeads,
    """
    model = model.cpu()
    model.eval()

    for name, module in model.named_children():
        if isinstance(module, nn.ModuleList):
            # Quantize feature_tokenizers or output_heads
            if name in ['feature_tokenizers', 'output_heads']:
                for i, submodule in enumerate(module):
                    module[i] = quantize_dynamic(submodule, {nn.Linear}, dtype=torch.qint8)
            else:
                print(f"[INFO] Skipping ModuleList: {name}")
        elif isinstance(module, nn.Linear):
            # If top-level Linear (rare)
            setattr(model, name, quantize_dynamic(module, {nn.Linear}, dtype=torch.qint8))
        else:
            # Transformer and others -> DO NOT QUANTIZE
            print(f"[INFO] Skipping {name} ({type(module)})")

    print("[INFO] Safe dynamic quantization complete.")
    return model


# Utility: Structured Pruning
def structured_prune_model(model, prune_amount=0.3, device="cuda"):
    model.to(device)
    total_weights = 0
    total_pruned = 0

    print(" Applying structured pruning...")

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Optional: Skip tiny heads/output layers
            if module.out_features <= 1:
                print(f" Skipping small layer: {name} (out_features={module.out_features})")
                continue

            # Apply structured pruning
            prune.ln_structured(module, name="weight", amount=prune_amount, n=2, dim=0)

            # Make pruning permanent
            prune.remove(module, 'weight')

            # Count pruned weights
            with torch.no_grad():
                weight = module.weight
                layer_total = weight.numel()
                layer_pruned = (weight == 0).sum().item()
                layer_sparsity = layer_pruned / layer_total

                total_weights += layer_total
                total_pruned += layer_pruned

                print(f" {name} → sparsity: {layer_sparsity:.2f}")

    if total_weights == 0:
        print(" Warning: No eligible layers were pruned!")
        return model

    pruned_percent = 100 * total_pruned / total_weights
    print(f" Total structured sparsity: {pruned_percent:.2f}%")

    return model


class PruningProblem(Problem):
    def __init__(self, model, val_loader, device="cuda"):
        super().__init__(n_var=1, n_obj=2, n_constr=0, xl=0.05, xu=0.5)  # n_obj=4 now
        self.model = deepcopy(model).to(device)
        self.val_loader = val_loader
        self.device = device

    def _evaluate(self, X, out, *args, **kwargs):
        results = []
        for prune_ratio in X:
            print(f"Evaluating pruning ratio: {prune_ratio[0]:.2f}")
            model_copy = deepcopy(self.model).to(self.device)
            pruned_model = structured_prune_model(model_copy, prune_ratio[0], device=self.device)

            # Latency
            sample_input = [next(iter(self.val_loader))[0].to(self.device) for _ in range(6)]
            latency = measure_latency(pruned_model, sample_input, self.device)

            # Count parameters
            param_count = count_model_parameters(pruned_model)

            results.append([param_count, latency])

        out["F"] = np.array(results)


def train_nsga_pymoo(model, val_loader, device="cuda", prune_ratio_from_agent_or_log=None):

    problem = PruningProblem(model, val_loader, device)

    # Clamp PPO ratio
    safe_seed = None
    if prune_ratio_from_agent_or_log is not None:
        safe_seed = float(np.clip(prune_ratio_from_agent_or_log, 0.05, 0.5))
        print(f"[INFO] Seeding NSGA-II with PPO-derived ratio: {safe_seed:.2f}")

        # Create initial population with PPO seed
        sampling = FloatRandomSampling()
        initial_pop = Population.new("X", np.vstack([
            np.array([[safe_seed]]),            # 1st individual: PPO seed
            sampling(problem, 149).get("X")     # rest: random
        ]))
    else:
        print("[INFO] No PPO seed provided. Running NSGA-II from scratch.")
        initial_pop = None  # Let NSGA-II sample normally

    algorithm = NSGA2(
        pop_size=150,
        sampling=initial_pop,
        crossover=SBX(prob=0.9, eta=20),
        mutation=PM(eta=25),
        eliminate_duplicates=True
    )

    res = minimize(problem, algorithm, ("n_gen", 30), verbose=True)

    # losses, sparsities, flops, latencies = res.F[:, 0], res.F[:, 1], res.F[:, 2], res.F[:, 3]
    param_counts, latencies = res.F[:, 0], res.F[:, 1]
    # losses, latencies = res.F[:, 0], res.F[:, 1]

    plt.figure(figsize=(10, 6))
    plt.scatter(latencies, param_counts, color='black', s=40, edgecolors='k', marker='o', label='Candidate Models')

    plt.xlabel("Inference Latency (ms)", fontsize=12)
    plt.ylabel("Number of Parameters", fontsize=12)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    plt.legend(fontsize=12, loc='best', frameon=False)
    plt.tight_layout()
    plt.savefig("nsga2_pareto_front_params_vs_latency.pdf", dpi=300, bbox_inches="tight")
    plt.show()

    # Apply Minimum Manhattan Distance (MMD) for selection
    ideal_point = np.min(res.F, axis=0)  # Best value for each objective
    mmd_scores = np.sum(np.abs(res.F - ideal_point), axis=1)  # L1 distance to ideal

    best_index = np.argmin(mmd_scores)
    best_solution = res.X[best_index]

    print(f"[INFO] Best solution found at index {best_index} using MMD with objectives:")

    return best_solution


# PPO-Based Pruning Environment
# Updated PruningEnv for gymnasium
class PruningEnv(gym.Env):
    def __init__(self, model, val_loader, device="cuda"):
        super(PruningEnv, self).__init__()
        self.model = deepcopy(model).to(device)
        self.val_loader = val_loader
        self.device = device
        self.layers = [m for m in self.model.modules() if isinstance(m, nn.Linear)]
        self.state = np.ones(len(self.layers))

        self.action_space = spaces.Discrete(len(self.layers))  # Select a layer to prune
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.layers),), dtype=np.float32)

    def step(self, action):
        prune_ratio = min(max(action / 10, 0.1), 0.7)
        print(f" PPO pruning layer {action} at {prune_ratio:.2f}")

        layer_to_prune = self.layers[action]

        # Skip already pruned layers
        if not hasattr(layer_to_prune, "_pruned"):
            prune.ln_structured(layer_to_prune, name="weight", amount=prune_ratio, n=2, dim=0)
            layer_to_prune._pruned = True
        else:
            print(f"️ Layer {action} already pruned. Skipping...")

        # Evaluate loss
        val_loss = evaluate_model(self.model, self.val_loader, self.device)

        # Measure compression
        total_params = sum(p.numel() for p in self.model.parameters())
        zero_params = sum((p == 0).sum().item() for p in self.model.parameters())
        # Measure latency
        latency = measure_latency(
            self.model,
            [torch.randn(1, 10).to(self.device) for _ in range(6)],
            self.device
        )

        # Evaluate after pruning
        val_loss = evaluate_model(self.model, self.val_loader, self.device)
        # flops = compute_flops(self.model)
        sample_input = [next(iter(self.val_loader))[0].to(self.device) for _ in range(6)]
        latency = measure_latency(self.model, sample_input, self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        zero_params = sum((p == 0).sum().item() for p in self.model.parameters())

        # Updated reward function (based on your info)
        l_loss = 0.5
        l_latency = 2.0

        # Normalize reward automatically
        reward = (
            -l_loss * val_loss +
            l_latency * (1.0 / (latency + 1e-6))
        )

        # Normalize reward magnitude
        latency_penalty = -np.log(latency + 1e-3)  # or use np.log
        reward += l_latency * latency_penalty

        reward /= 10.0  # optional scaling, prevents very large/small rewards

        print(f"[Reward] {reward:.4f}")

        print(f" Final Reward: {reward:.4f}")

        self.state[action] = prune_ratio
        done = np.all(self.state < 1)

        return np.array(self.state, dtype=np.float32), reward, done, False, {}

    def reset(self, seed=None, options=None):  # Gymnasium requires `seed` and `options`
        self.model = deepcopy(self.model).to(self.device)
        self.state = np.ones(len(self.layers))
        return np.array(self.state, dtype=np.float32), {}  # Gymnasium returns state + info dictionary


def remove_weight_norm_all(model):
    """Remove weight normalization from all layers."""
    for name, module in model.named_modules():
        if hasattr(module, 'weight_g'):
            try:
                parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
                print(f" Removed weight normalization from: {name}")
            except Exception as e:
                print(f" Failed to remove weight norm from {name}: {e}")
    return model  # Ensure the model is returned


def apply_hybrid_pruning(env_class, model, val_loader, device, debug_mode=False):
    """Applies hybrid pruning with an option to skip PPO/NSGA-II for fast debugging."""

    # Ensure model is not None
    if model is None:
        raise ValueError("ERROR: Model is None before pruning!")

    # Move model to device
    model.to(device)
    dummy_env = DummyVecEnv([lambda: env_class(model, val_loader, device=device)])
    vec_env = VecNormalize(dummy_env, norm_obs=True, norm_reward=False)
    policy_kwargs = dict(
        features_extractor_class=CustomMLPExtractor,  # defined elsewhere
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[dict(pi=[128, 128], vf=[128, 128])]  # actor-critic separation
    )

    if debug_mode:
        print("Debug Mode: Skipping PPO & NSGA-II, using fixed pruning (30%)")
        prune_amount = 0.4
    else:
        # PPO-Based Pruning
        ppo_env = PruningEnv(model, val_loader, device)

        ppo_agent = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=3e-5,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.001,
            vf_coef=0.5,
            max_grad_norm=0.5,
            # policy_kwargs=policy_kwargs,
            verbose=1,
            device=device,
            tensorboard_log="./ppo_tensorboard/"
        )

        checkpoint_callback = CheckpointCallback(save_freq=1, save_path="./ppo_checkpoints/",
                                                 name_prefix="ppo_pruning")
        ppo_agent.learn(total_timesteps=2, callback=checkpoint_callback)

        def extract_best_prune_ratio_from_agent(agent, vec_env):
            """
            Extracts the best pruning ratio from the PPO agent's value function.
            Uses the environment's reset state as input and selects the best action.
            """
            obs = vec_env.reset()

            # Get the best action from the agent
            action, _ = agent.predict(obs, deterministic=True)

            # Convert to pruning ratio in [0.05, 0.5]
            prune_ratio = float(action[0]) / 10.0
            prune_ratio = float(np.clip(prune_ratio, 0.05, 0.5))  # Clamp safely

            print(f"[INFO] Extracted prune ratio from PPO agent: {prune_ratio:.2f}")
            return prune_ratio

        ppo_prune_ratio = extract_best_prune_ratio_from_agent(ppo_agent, dummy_env)
        # NSGA-II Pruning
        print(f"Model before pruning: {model}")
        print("Applying NSGA-II pruning...")
        best_nsga_solution = train_nsga_pymoo(model, val_loader, device, prune_ratio_from_agent_or_log=ppo_prune_ratio)

        if best_nsga_solution is None:
            print("ERROR: NSGA-II returned None! Using fixed pruning (30%).")
            prune_amount = 0.4
        else:
            prune_amount = min(0.2, best_nsga_solution[0])

    # Apply structured pruning
    print(f"Applying structured pruning with amount: {prune_amount}")
    model = structured_prune_model(model, prune_amount=prune_amount, device=device)
    if model is None:
        raise ValueError("ERROR: Pruning failed! Model is None.")

    # Fine-tuning pruned model
    print("Fine-tuning pruned model...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.SmoothL1Loss()

    for epoch in range(1):
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            for param in model.parameters():
                param.requires_grad = True

            optimizer.zero_grad()
            batch_X_list = [batch_X.clone() for _ in range(len(model.feature_tokenizers))]
            outputs = model(batch_X_list)
            loss = sum(loss_fn(p, batch_y[:, i:i + 1]) for i, p in enumerate(outputs))
            loss.backward()
            optimizer.step()

    print("Fine-tuning complete.")

    # Apply dynamic quantization with debugging
    print("[INFO] Attempting dynamic quantization...")
    model.eval().cpu()  # Move model to CPU for quantization
    for name, module in model.named_modules():
        if hasattr(module, 'weight_g'):
            try:
                parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
                print(f"[DEBUG] Removed weight norm from: {name}")
            except Exception as e:
                print(f"[WARN] Could not remove weight norm from {name}: {e}")

    try:
        quantized_model = apply_safe_dynamic_quantization(model)
        print("[INFO] Quantization successful. Moving quantized model to CPU.")
        quantized_model.to("cpu")  # ensure model stays on CPU for inference
    except Exception as e:
        print(f"[ERROR] Quantization failed: {e}. Using original model.")
        quantized_model = model.cpu()

    return quantized_model
