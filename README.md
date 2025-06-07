# Calibrated LLM-Guided Reinforcement Learning

This repository contains all code, experiments, and evaluations for our paper:

> **Policy Shaping with Uncertainty-Aware LLM for Multi-Task Reinforcement Learning**

We explore how calibrated Large Language Models (LLMs) can guide reinforcement learning agents in sparse, multi-task environments using uncertainty-aware policy shaping. Our implementation features


## 📁 Project Structure

```bash
.
├── A-Astrick/        # PPO + Calibrated LLM  experiments
├── DQN/             # DQN  baseline with Dueling DQN + PER
├── Qlearning/       # Oracle-based Q-learning + LLM fine-tuning
├── charts          # Project Result Charts
└── ...

🧪 Experimental Pipelines

🔷 asterisk/ – PPO + Calibrated LLM
This folder contains experiments with PPO guided by LLMs on a 4x4 MiniGrid. It supports different policy shaping strategies.

Run from terminal:

cd A-Astrick/ 
python main.py --mode all                # Run all experiments (default: 4x4 calibrated)
python main.py --mode 4x4                # Calibrated (main experiment)
python main.py --mode baseline4x4        # Unguided PPO baseline
python main.py --mode uncalibrated4x4    # Uncalibrated LLM guidance
python main.py --mode linear4x4          # Linear-decay shaping


🔷 DQN/  – DQN
This folder implements a Deep Q-Network variant with:

Run the experiment:
cd DQN/ 
python main.py


🔷 Qlearning/ – Q-learning Oracle + LLM Fine-tuning 
This folder uses a Q-learning Oracle to generate labeled trajectories, fine-tunes a BERT-based LLM using this data, and runs PPO guided by the calibrated LLM.

cd Qlearning/
python main.py

Install the required dependencies:
pip install torch gym pandas transformers tqdm

Tested on:
✅ Python ≥ 3.8
✅ PyTorch ≥ 1.12
✅ Works on Linux, macOS

The reward training curves for different models (Calibrated LLM RL, Uncalibrated RL, Unguided PPO, Q-learning, DQN, etc.) are stored in CSV format and can be visualized using the provided plot.py script in the charts/ directory.
