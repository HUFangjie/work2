#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SpecShield 消融二维散点图绘制脚本
=================================

读取 `specshield_ablation.py` 生成的 plot_data/*.npz，
绘制三联图：
  (a) Raw parameter / raw-difference features
  (b) DCT-based features
  (c) DWT-based features

输出：
- Nature 风格 PNG
- Nature 风格 PDF
"""

import argparse
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


VARIANT_META: Dict[str, Tuple[str, str]] = {
    "raw_diff": ("(a) Raw parameter / raw-difference features", "raw_diff_best_projection.npz"),
    "dct": ("(b) DCT-based features", "dct_best_projection.npz"),
    "dwt": ("(c) DWT-based features", "dwt_best_projection.npz"),
}


def setup_nature_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_projection(path: str):
    data = np.load(path, allow_pickle=True)
    coords = data["coords"]
    role_labels = data["role_labels"]
    epoch = int(data["epoch"][0]) if "epoch" in data else -1
    tpr = float(data["tpr"][0]) * 100 if np.max(data["tpr"]) <= 1.0 else float(data["tpr"][0])
    fpr = float(data["fpr"][0]) * 100 if np.max(data["fpr"]) <= 1.0 else float(data["fpr"][0])
    return coords, role_labels, epoch, tpr, fpr


def plot_ablation_figure(plot_data_dir: str, output_dir: str):
    setup_nature_style()
    os.makedirs(output_dir, exist_ok=True)

    benign_color = "#3C78D8"
    malicious_color = "#D1495B"

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2))

    for ax, (variant_key, (title, filename)) in zip(axes, VARIANT_META.items()):
        path = os.path.join(plot_data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到绘图数据文件: {path}")

        coords, role_labels, epoch, tpr, fpr = load_projection(path)
        benign_mask = role_labels == "benign"
        malicious_mask = role_labels == "malicious"

        ax.scatter(
            coords[benign_mask, 0], coords[benign_mask, 1],
            s=28, c=benign_color, alpha=0.85, edgecolors="white", linewidths=0.35, label="Benign clients"
        )
        ax.scatter(
            coords[malicious_mask, 0], coords[malicious_mask, 1],
            s=34, c=malicious_color, alpha=0.92, edgecolors="white", linewidths=0.35, label="Malicious clients"
        )

        if benign_mask.any():
            benign_center = coords[benign_mask].mean(axis=0)
            ax.scatter(benign_center[0], benign_center[1], marker="X", s=80, c=benign_color, edgecolors="black", linewidths=0.4)
        if malicious_mask.any():
            malicious_center = coords[malicious_mask].mean(axis=0)
            ax.scatter(malicious_center[0], malicious_center[1], marker="X", s=90, c=malicious_color, edgecolors="black", linewidths=0.4)

        ax.set_title(title, pad=10, fontweight="bold")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            0.02, 0.98,
            f"Best epoch: {epoch}\nTPR: {tpr:.1f}%\nFPR: {fpr:.1f}%",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#CCCCCC", alpha=0.9)
        )

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=[0, 0, 1, 0.95], w_pad=2.0)

    png_path = os.path.join(output_dir, "specshield_ablation_feature_scatter.png")
    pdf_path = os.path.join(output_dir, "specshield_ablation_feature_scatter.pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"散点图已保存: {png_path}")
    print(f"散点图已保存: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="绘制 SpecShield 消融二维散点图")
    parser.add_argument("--plot-data-dir", default="ablation_results_cifar10_fang/plot_data")
    parser.add_argument("--output-dir", default="ablation_results_cifar10_fang/figures")
    args = parser.parse_args()
    plot_ablation_figure(args.plot_data_dir, args.output_dir)


if __name__ == "__main__":
    main()
