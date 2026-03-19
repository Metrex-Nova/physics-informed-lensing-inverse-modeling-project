# Report: Physics-Informed Gravitational Lensing Inverse Modeling

This report documents the methodology, implementation, and results for a physics-informed inverse modeling pipeline that reconstructs a mass convergence map (κ) from a lensed image.

## 1. Problem statement

Given an observed lensed image, the goal is to recover the lens convergence κ(x) that produced it. The forward physics is:

1. **Poisson equation** (projected mass → potential)

   $$\nabla^2 \psi(\mathbf{x}) = \kappa(\mathbf{x})$$

2. **Deflection field**

   $$\boldsymbol{\alpha}(\mathbf{x}) = \nabla \psi(\mathbf{x})$$

3. **Lens equation** (ray tracing)

   $$\mathbf{x}_s = \mathbf{x}_i - \boldsymbol{\alpha}(\mathbf{x}_i)$$

The inverse learning task is to map from the lensed image back to κ, while enforcing consistency with the lens equation.

## 2. Implementation details

### Dataset and simulation

- **Mass profiles**: Singularity Isothermal Sphere (SIS) and a simplified NFW-like radial profile.
- **Source model**: random combination of Gaussian blobs.
- **Lensed image generation**: given κ and source, compute ψ via FFT Poisson solver, then compute α and apply grid sampling to obtain lensed image.

### Physics modules

- **Poisson solver**: spectral solve of ∇²ψ = κ using `torch.fft`.
- **Deflection**: compute gradient in Fourier space, producing α(x). The output is shaped `(B, 2, H, W)`.
- **Lens equation**: implement ray tracing using `torch.nn.functional.grid_sample`.

### Models

- **Baseline CNN**: 4 conv layers predicting κ directly.
- **U-Net**: standard encoder-decoder with skip connections.

### Loss functions

- **Baseline**: MSE(κ_pred, κ_true)
- **Physics-informed**: MSE(κ_pred, κ_true) + λ · MSE(lensed_pred, lensed_obs)

## 3. Experiments

The repository includes scripts to compare baseline and physics-informed models and to test robustness to input noise.

- `compare_models.py`: computes κ and lensed-image errors for both models.
- `noise_robustness.py`: evaluates performance at different input noise levels.

## 4. Results

The physics-informed model is evaluated by how well it reproduces both the true mass map (κ) and the observed lensed image. The loss term on the re-lensed image enforces physical consistency.

## 5. Limitations and future work

- Current mass profiles are simplified; real lenses require elliptical profiles and multiple components.
- The source model is synthetic and lacks realistic galaxy morphology.
- Extending to real survey data requires PSF modeling, noise modeling, and calibration.

## 6. How to run

Use:

```bash
python main.py --mode train_physics
```

Additional scripts:

- `python main.py --mode train_baseline`
- `python main.py --mode compare`
- `python main.py --mode noise`

---

*This report is part of the Physics-Informed Gravitational Lensing Inverse Modeling project.*
