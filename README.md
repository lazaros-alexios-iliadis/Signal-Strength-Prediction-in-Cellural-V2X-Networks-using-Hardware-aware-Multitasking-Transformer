# Signal-Strength-Prediction-in-Cellural-V2X-Networks-using-Hardware-aware-Multitasking-Transformer

This repository contains the code accompanying our work on hardware-aware multitask learning for V2X radio link quality prediction.
We propose a hybrid pruning strategy that combines reinforcement learning (PPO) and evolutionary optimization (NSGA-II) with structured pruning and dynamic quantization to deploy efficient multitask Transformers on resource-constrained vehicular hardware.

# Problem Statement

Vehicular networks (V2X) require fast and accurate prediction of radio link quality metrics (e.g., RSRP, RSRQ, SNR).
Deep learning methods achieve high accuracy, but deployment on edge devices faces constraints in latency, energy consumption, and memory. We reformulate the problem as multitask regression and introduce a hybrid pruning + quantization pipeline for efficiency-aware model compression.

# Method Overview

Model: MultiTaskFTTransformer (Transformer_model.py)
Hybrid Pruning (hybrid_pruning.py)
Baselines (pruning_methods.py)

# Requirements

Python 3.9+
PyTorch 2.0+
scikit-learn
stable-baselines3
pymoo
matplotlib, seaborn
