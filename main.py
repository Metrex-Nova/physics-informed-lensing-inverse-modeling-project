import argparse

from lensing.training import train_baseline, train_physics_informed
from lensing.experiments import compare_models, noise_robustness


def main():
    parser = argparse.ArgumentParser(description="Physics-Informed Gravitational Lensing Inverse Modeling")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train_baseline", "train_physics", "compare", "noise"],
        default="train_physics",
        help="Mode to run: train baseline, physics-informed training, compare models, or noise robustness.",
    )

    args = parser.parse_args()

    if args.mode == "train_baseline":
        train_baseline.run()
    elif args.mode == "train_physics":
        train_physics_informed.run()
    elif args.mode == "compare":
        compare_models.run()
    elif args.mode == "noise":
        noise_robustness.run()
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
