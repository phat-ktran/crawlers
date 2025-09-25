#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot loss vs steps from CSV file")
    parser.add_argument("--input", required=True, help="Path to input text/CSV file")
    parser.add_argument("--output", required=True, help="Path to save output plot (e.g., loss.png)")
    parser.add_argument("--smooth", type=int, default=0,
                        help="Apply rolling average with given window size (default: 0 = no smoothing)")
    args = parser.parse_args()

    # Read the CSV
    df = pd.read_csv(args.input)

    if not {"step", "loss"}.issubset(df.columns):
        raise ValueError("Input file must have 'step' and 'loss' columns")

    # Apply smoothing if requested
    if args.smooth > 1:
        df["loss"] = df["loss"].rolling(window=args.smooth, min_periods=1).mean()

    # Plot style
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(10, 6))

    # Draw line (no markers to avoid density issues)
    plt.plot(df["step"], df["loss"], linewidth=2, color="tab:blue", label="Loss")

    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Loss vs Step", fontsize=14, weight="bold")
    plt.legend()
    plt.grid(alpha=0.3)

    # Save
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"✅ Plot saved to {args.output}")


if __name__ == "__main__":
    main()
