# specifically to recreate figure 5
# run "python plot_figure5.py" to print a similar figure to figure 5 in the paper P46 from the learning curve data in the folder learning_curves.
# if you want to print data from a different folder, it has to be manually change in the code
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_figure5(learning_curves_dir=None, output_path=None, show=True):
    # change folder her or in the functional argument
    root = Path(learning_curves_dir) if learning_curves_dir else (
        Path(__file__).resolve().parent / "learning_curves"
    )
    if not root.is_dir():
        raise FileNotFoundError(f"Learning-curves directory not found: {root}")

    env_order = [
        "Ant",
        "HalfCheetah",
        "Hopper",
        "InvertedDoublePendulum",
        "InvertedPendulum",
        "Reacher",
        "Walker",
    ]
    env_dirs = [root / name for name in env_order if (root / name).is_dir()]
    if not env_dirs:
        raise FileNotFoundError(f"No environment folders found in: {root}")

    columns = 3
    rows = ceil(len(env_dirs) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(4.5 * columns, 3.2 * rows), squeeze=False
    )
    timesteps = np.arange(1, 201) * 5_000

    for ax, env_dir in zip(axes.flat, env_dirs):
        files = sorted(env_dir.glob("TD3_*.npy"))
        if not files:
            raise FileNotFoundError(f"No TD3 .npy files found in: {env_dir}")

        runs = []
        for path in files:
            curve = np.load(path, allow_pickle=False)
            if curve.shape != (201,):
                raise ValueError(
                    f"Expected {path} to contain 201 evaluations; got {curve.shape}"
                )
            runs.append(curve[1:])  # Skip the initial random-policy evaluation.

        runs = np.stack(runs)
        mean = runs.mean(axis=0)
        std = runs.std(axis=0)
        title = "Walker2d" if env_dir.name == "Walker" else env_dir.name

        ax.plot(timesteps, mean, color="C0")
        ax.fill_between(timesteps, mean - std, mean + std, color="C0", alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Average return")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
        ax.grid(True, alpha=0.25)

    for ax in axes.flat[len(env_dirs):]:
        ax.set_visible(False)

    fig.tight_layout()
    # save png
    if output_path is None:
        output_path = root.parent / "figure5.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes


if __name__ == "__main__":
    plot_figure5()