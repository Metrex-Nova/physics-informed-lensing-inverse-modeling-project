# Physics-Informed Gravitational Lensing Inverse Modeling

This repo implements a **physics-informed machine learning pipeline** to reconstruct a **mass density map (convergence `κ`)** from a **gravitationally lensed image**. The approach combines:

- **Synthetic lensing simulation** (SIS & NFW lens models)
- **FFT-based Poisson solve** (∇²ψ = κ) in the spectral domain
- **Differentiable lensing pipeline** (ψ → α → lens equation)
- **Deep learning models** (baseline CNN + U-Net)
- **Physics-informed loss** that enforces consistency with the lens equation

---

## ✅ Quickstart

1. Create a Python environment and install dependencies:

```bash
python -m venv ./venv
# Windows
./venv/Scripts/activate
# macOS / Linux
# source ./venv/bin/activate

pip install -r requirements.txt
```

2. Run the main pipeline:

```bash
python main.py --mode train_physics
```

3. Compare results:

```bash
python main.py --mode compare
```

---

## 📁 Project Structure

- `main.py` — entry point (runs training/experiments)
- `lensing/` — core physics, data generation, models, training, utilities
- `plots/` — exported figures
- `checkpoints/` — saved model weights

---

## 🧠 What’s implemented

### Physics
- **Mass profiles**: SIS and simplified NFW
- **Poisson solver**: FFT-based solve of `∇²ψ = κ`
- **Deflection**: `α = ∇ψ`
- **Lens equation**: `x_source = x_image - α(x_image)`

### Models
- **Baseline CNN** (image-to-image mapping)
- **U-Net** (encoder-decoder with skip connections)

### Training
- Baseline training with MSE on `κ`
- Physics-informed training with combined loss:
  - MSE on `κ`
  - λ · MSE on reconstructed lensed image

---

## 📌 Notes
- The pipeline is modular: you can swap mass models, noise levels, and network architectures.
- Outputs are saved under `plots/` and `checkpoints/`.

---

## 🧪 Running Experiments
- `compare_models.py` compares baseline vs physics-informed models.
- `noise_robustness.py` evaluates robustness to input noise.

---

If you want to extend this for real survey data or more sophisticated lens profiles, the modular structure makes it easy to plug in new physics components.
