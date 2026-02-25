```markdown
# JetBot Sim2Real: Safe Reinforcement Learning in Low-Friction Environments

This repository contains the code for an EPFL semester project conducted in the Sycamore Lab. The goal of this project is to bridge the Sim2Real gap for a differential-drive JetBot navigating a maze under challenging, low-friction conditions (e.g., puddles) using Safe Reinforcement Learning (PPO Lagrangian) implemented in **JAX/Flax**.

This codebase provides a highly optimized, fully vectorized JAX environment—replicated from an IsaacLab setup—and tracks the evolution of the control policy across three distinct research phases to overcome static friction (stiction) and domain shift.

## 📂 Repository Structure

The project is structured chronologically, reflecting the research steps taken to solve the Sim2Real gap:

* **`/Phase_1_MLP`**: The baseline implementation. Uses a standard Feedforward network (MLP) and Uniform Domain Randomization (UDR) for the motor deadzone parameters.
* **`/Phase_2_FrameStacking`**: Introduces finite memory via Frame Stacking ($k=3$) to infer velocity, and replaces UDR with a Gaussian Mixture Model (GMM) based on empirical system identification to model "Puddle" vs. "Dry" states.
* **`/Phase_3_LSTM`**: The final, most advanced pipeline. Replaces Frame Stacking with an LSTM recurrent policy for infinite memory. Replaces mathematical deadzone formulas with a Data-Driven empirical lookup table (`physics_lookup.npy`) and introduces an active "Dance" (stiction-breaking) logic.
* **`/data_processing`**: Contains the raw system identification data collected from the real JetBot via OptiTrack, as well as the scripts used to extract the GMM parameters (Phase 2) and generate the empirical lookup table (Phase 3).

## 🚀 Getting Started

### Prerequisites
It is recommended to run this code on a machine with a GPU or TPU for JAX acceleration.

```bash
# Clone the repository
git clone [https://github.com/ftanguy/JetBot-Sim2Real.git](https://github.com/ftanguy/JetBot-Sim2Real.git)
cd JetBot-Sim2Real

# Install dependencies (ensure you have the correct JAX version for your hardware)
pip install -r requirements.txt

```

### Running the Code

Each phase is completely self-contained. To train an agent, navigate to the desired phase and run the training script.

```bash
cd Phase_3_LSTM

# Start PPO training
python train_ppo_lagrangian.py

```

Training will automatically generate a `plots/` folder containing training metrics (Reward, Average Cost, Lagrange Multiplier $\lambda$) and save the final network weights as `trained_params.pkl`.

### Evaluating and Plotting

To evaluate a trained policy and plot the resulting trajectories on the maze map:

```bash
python play_and_plot_trajectories.py

```

*(Note: Ensure `trained_params.pkl` is present in the directory before running the evaluation).*

## 🧠 Key Features & Technical Details

* **100% JAX-Native:** The environment, physics steps, PPO algorithm, and Lagrangian dual-updates are all JAX-jitted and heavily parallelized (e.g., running 4096 environments simultaneously).
* **Safe RL (Lagrangian):** The agent is penalized for getting too close to walls. The Lagrange multiplier dynamically adjusts during training to strictly enforce the safety constraint.
* **Data-Driven Physics:** In Phase 3, standard kinematic equations are augmented by sampling actual speeds from a binned lookup table of real-world OptiTrack data to perfectly capture low-level non-linearities like deadzones and stiction.

## 🙏 Acknowledgements

This project was developed at the **EPFL Sycamore Lab**. Special thanks to Kai Ren and Tingting Ni for their supervision, and to Federico for the original IsaacLab implementation that served as the reference for this JAX environment.
