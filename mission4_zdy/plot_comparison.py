"""Create the task 4 one-shot versus three-shot summary figure."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
CONDITIONS = ("1shot", "3shot")
METRICS = (
    ("relation_test_accuracy", "Relation test accuracy (%)", (98.5, 99.2)),
    ("digit_test_accuracy", "Digit test accuracy (%)", (94.5, 98.0)),
)
COLORS = ("#2563eb", "#0f9f8f")


def main():
    summaries = {
        condition: json.loads(
            (ROOT / condition / "training_results" / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        for condition in CONDITIONS
    }
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for axis, (metric, title, limits) in zip(axes, METRICS):
        for position, condition in enumerate(CONDITIONS):
            stats = summaries[condition]["metrics"][metric]
            axis.errorbar(
                position,
                stats["mean"],
                yerr=stats["std"],
                fmt="o",
                markersize=9,
                capsize=7,
                elinewidth=2,
                color=COLORS[position],
                label=condition,
            )
            axis.annotate(
                f'{stats["mean"]:.3f} ± {stats["std"]:.3f}',
                (position, stats["mean"]),
                xytext=(0, 13),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        axis.set_xticks(range(2), ["1-shot", "3-shot"])
        axis.set_xlim(-0.28, 1.28)
        axis.set_ylim(*limits)
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.set_title(title)
    figure.suptitle("Siamese: 1-shot vs 3-shot (5 seeds; mean ± sample SD)")
    figure.tight_layout()
    figure.savefig(ROOT / "comparison_1shot_3shot.svg", bbox_inches="tight")
    figure.savefig(ROOT / "comparison_1shot_3shot.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
