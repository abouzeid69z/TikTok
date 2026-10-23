"""Generate square, LinkedIn-ready summary cards from verified project results."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tiktok-matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "linkedin"
BACKGROUND = "#0f172a"
CARD = "#1e293b"
PRIMARY = "#38bdf8"
SECONDARY = "#34d399"
ACCENT = "#f59e0b"
TEXT = "#f8fafc"
MUTED = "#cbd5e1"


def canvas() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=150)
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def rounded_card(ax: plt.Axes, x: float, y: float, width: float, height: float) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=1,
            edgecolor="#334155",
            facecolor=CARD,
        )
    )


def footer(ax: plt.Axes, page: str) -> None:
    ax.text(0.06, 0.045, "github.com/abouzeid69z/TikTok", color=PRIMARY, fontsize=10.5)
    ax.text(0.94, 0.045, page, color=MUTED, fontsize=10.5, ha="right")


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUTPUT_DIR / name, facecolor=BACKGROUND, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def create_cover(metrics: dict) -> None:
    fig, ax = canvas()
    ax.text(0.06, 0.92, "DATA ANALYTICS PORTFOLIO", color=PRIMARY, fontsize=12, weight="bold")
    ax.text(0.06, 0.82, "TikTok Claim\nClassification", color=TEXT, fontsize=31, weight="bold", va="top")
    ax.text(
        0.06,
        0.58,
        "Built through the Google Advanced Data Analytics\nProfessional Certificate",
        color=MUTED,
        fontsize=12.5,
        linespacing=1.4,
        va="top",
    )

    stats = [
        (f"{metrics['data']['raw_rows']:,}", "videos analyzed"),
        ("4", "models compared"),
        (f"{metrics['test_metrics']['f1']:.2%}", "test F1 score"),
    ]
    for index, (value, label) in enumerate(stats):
        x = 0.06 + index * 0.305
        rounded_card(ax, x, 0.30, 0.27, 0.16)
        ax.text(x + 0.135, 0.39, value, color=SECONDARY, fontsize=23, weight="bold", ha="center")
        ax.text(x + 0.135, 0.335, label, color=MUTED, fontsize=10.5, ha="center")

    ax.text(0.06, 0.235, "Python  •  Statistics  •  Machine Learning  •  Data Visualization", color=TEXT, fontsize=11.5)
    ax.text(
        0.06,
        0.195,
        "Goal: help moderation teams prioritize videos containing\nclaims for human review.",
        color=MUTED,
        fontsize=13,
        linespacing=1.5,
        va="top",
    )
    footer(ax, "1 / 3")
    save(fig, "01_project_overview.png")


def create_findings(data: pd.DataFrame) -> None:
    fig, ax = canvas()
    ax.text(0.06, 0.92, "KEY FINDINGS", color=PRIMARY, fontsize=12, weight="bold")
    ax.text(0.06, 0.85, "Claims and opinions show\nvery different engagement", color=TEXT, fontsize=24, weight="bold", va="top")

    medians = data.groupby("claim_status")["video_view_count"].median()
    claim_views = float(medians["claim"])
    opinion_views = float(medians["opinion"])
    ratio = claim_views / opinion_views

    rounded_card(ax, 0.06, 0.48, 0.88, 0.22)
    ax.text(0.11, 0.63, "Median video views", color=MUTED, fontsize=12)
    ax.text(0.11, 0.555, f"{claim_views:,.0f}", color=SECONDARY, fontsize=28, weight="bold")
    ax.text(0.11, 0.515, "CLAIM", color=MUTED, fontsize=10, weight="bold")
    ax.text(0.54, 0.555, f"{opinion_views:,.0f}", color=ACCENT, fontsize=28, weight="bold")
    ax.text(0.54, 0.515, "OPINION", color=MUTED, fontsize=10, weight="bold")

    rounded_card(ax, 0.06, 0.24, 0.42, 0.17)
    ax.text(0.10, 0.345, f"{ratio:.0f}×", color=PRIMARY, fontsize=27, weight="bold")
    ax.text(0.10, 0.275, "higher median views\nfor claims", color=MUTED, fontsize=10.5, linespacing=1.35)

    rounded_card(ax, 0.52, 0.24, 0.42, 0.17)
    claim_share = (data["claim_status"] == "claim").mean()
    ax.text(0.56, 0.345, f"{claim_share:.1%}", color=PRIMARY, fontsize=27, weight="bold")
    ax.text(0.56, 0.275, "of cleaned videos\nare labeled claims", color=MUTED, fontsize=10.5, linespacing=1.35)

    ax.text(0.06, 0.135, "Educational dataset: association does not establish causation.", color=MUTED, fontsize=10)
    footer(ax, "2 / 3")
    save(fig, "02_key_findings.png")


def create_results(metrics: dict) -> None:
    fig, ax = canvas()
    ax.text(0.06, 0.92, "MODEL RESULTS", color=PRIMARY, fontsize=12, weight="bold")
    ax.text(0.06, 0.85, "Random Forest selected\non validation F1", color=TEXT, fontsize=27, weight="bold", va="top")

    values = [
        ("Accuracy", metrics["test_metrics"]["accuracy"]),
        ("Precision", metrics["test_metrics"]["precision"]),
        ("Recall", metrics["test_metrics"]["recall"]),
        ("F1 score", metrics["test_metrics"]["f1"]),
    ]
    positions = [(0.06, 0.49), (0.52, 0.49), (0.06, 0.29), (0.52, 0.29)]
    for (label, value), (x, y) in zip(values, positions):
        rounded_card(ax, x, y, 0.42, 0.15)
        ax.text(x + 0.04, y + 0.086, f"{value:.2%}", color=SECONDARY, fontsize=24, weight="bold")
        ax.text(x + 0.04, y + 0.035, label, color=MUTED, fontsize=11)

    report = metrics["test_classification_report"]
    missed = int(report["claim"]["support"] * (1 - report["claim"]["recall"]) + 0.5)
    total = int(report["macro avg"]["support"])
    correct = round(metrics["test_metrics"]["accuracy"] * total)
    ax.text(0.06, 0.195, f"{correct:,} of {total:,} test videos classified correctly", color=TEXT, fontsize=13, weight="bold")
    ax.text(0.06, 0.145, f"Only {missed} claims missed • No opinions incorrectly flagged", color=MUTED, fontsize=11.5)
    footer(ax, "3 / 3")
    save(fig, "03_model_results.png")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
    data = pd.read_csv(ROOT / "data" / "tiktok_dataset.csv").dropna()
    create_cover(metrics)
    create_findings(data)
    create_results(metrics)
    print(f"Created three LinkedIn images in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
