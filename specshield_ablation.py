#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SpecShield 消融实验脚本
=================================

三个方案：
A. Full SpecShield (FGSM + DWT + 聚类)
B. DCT Variant     (FGSM + DCT + 聚类)
C. Raw-Diff Variant(FGSM + Raw Feature Difference + 聚类)

说明：
- 为避免影响原始框架，本文件独立实现消融流程。
- 依赖 `optimized_fl_framework.py` 与 `wavelet_defense.py` 中已有的数据、模型、
  攻击与训练辅助工具。
- 会额外保存绘图散点所需的二维投影数据到单独目录，便于后续独立调图。
"""

import argparse
import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.fftpack import dct
from sklearn.decomposition import PCA

from optimized_fl_framework import (
    FLConfig,
    DataManager,
    ModelFactory,
    FederatedLearning,
    AttackFactory,
    DefenseEvaluator,
    StepLRScheduler,
)
from wavelet_defense import WaveletDefense


@dataclass
class AblationResult:
    scheme: str
    best_epoch: int
    best_test_acc: float
    best_val_acc: float
    tpr: float
    fpr: float
    plot_data_path: str


class ProjectionMixin:
    """为不同特征方案提供统一的二维投影接口。"""

    def compute_feature_projection(
        self, original_updates: torch.Tensor, probed_updates: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        feature_vectors = self.extract_feature_differences(original_updates, probed_updates)
        if len(feature_vectors) == 0:
            return np.zeros((0, 2)), np.zeros((0, 0))

        try:
            normalized = self.scaler.fit_transform(feature_vectors)
        except Exception:
            feature_vectors = np.asarray(feature_vectors)
            feature_max = np.max(feature_vectors, axis=0)
            feature_min = np.min(feature_vectors, axis=0)
            ranges = feature_max - feature_min
            ranges[ranges == 0] = 1
            normalized = (feature_vectors - feature_min) / ranges

        normalized = np.clip(normalized, -5, 5)
        n_components = min(2, normalized.shape[0], normalized.shape[1])
        if n_components < 1:
            return np.zeros((len(normalized), 2)), normalized

        coords = PCA(n_components=n_components).fit_transform(normalized)
        if n_components == 1:
            coords_2d = np.zeros((coords.shape[0], 2))
            coords_2d[:, 0] = coords[:, 0]
            return coords_2d, normalized
        return coords, normalized


class DWTAblationDefense(ProjectionMixin, WaveletDefense):
    """原始 SpecShield：DWT 特征。"""


class DCTAblationDefense(ProjectionMixin, WaveletDefense):
    """DCT 消融版本。"""

    def extract_feature_differences(self, original_updates, probed_updates):
        features = []
        n_chunks = 16

        for orig_update, probed_update in zip(original_updates, probed_updates):
            diff = (probed_update - orig_update).detach().cpu().numpy()
            coeffs = dct(diff, norm="ortho")
            chunks = np.array_split(coeffs, n_chunks)

            vector = []
            for chunk in chunks:
                abs_chunk = np.abs(chunk)
                vector.extend([
                    float(np.mean(abs_chunk)),
                    float(np.std(chunk)),
                    float(np.max(abs_chunk)),
                    float(np.sum(chunk ** 2) / max(len(chunk), 1)),
                ])
            features.append(np.asarray(vector, dtype=np.float32))

        return np.asarray(features)


class RawDiffAblationDefense(ProjectionMixin, WaveletDefense):
    """不做频域变换，直接使用 raw-difference 特征。"""

    def extract_feature_differences(self, original_updates, probed_updates):
        features = []
        n_chunks = 32

        for orig_update, probed_update in zip(original_updates, probed_updates):
            diff = torch.abs(probed_update - orig_update).detach().cpu().numpy()
            chunks = np.array_split(diff, n_chunks)

            vector = []
            for chunk in chunks:
                vector.extend([
                    float(np.mean(chunk)),
                    float(np.std(chunk)),
                    float(np.max(chunk)),
                ])
            features.append(np.asarray(vector, dtype=np.float32))

        return np.asarray(features)


class SpecShieldAblationRunner(FederatedLearning):
    """独立的消融实验运行器。"""

    VARIANT_MAP = {
        "raw_diff": ("Raw-Diff Variant", RawDiffAblationDefense),
        "dct": ("DCT Variant", DCTAblationDefense),
        "dwt": ("Full SpecShield", DWTAblationDefense),
    }

    def __init__(self, config: FLConfig, ablation_root: str):
        super().__init__(config)
        self.ablation_root = ablation_root
        self.plot_data_dir = os.path.join(ablation_root, "plot_data")
        os.makedirs(self.plot_data_dir, exist_ok=True)

    def _create_ablation_defender(
        self, variant_key: str, model: nn.Module, nusers: int, input_shape: Tuple[int, ...]
    ) -> WaveletDefense:
        defender_cls = self.VARIANT_MAP[variant_key][1]
        return defender_cls(
            model,
            nusers,
            input_shape=input_shape,
            device=self.device,
            logger=self.logger,
            verbose=self.verbose_logging,
        )

    def _evaluate_model(
        self, model: nn.Module, criterion: nn.Module, data_tensor: torch.Tensor, label_tensor: torch.Tensor,
        batch_size: int = 128
    ) -> Tuple[float, float]:
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total = 0

        with torch.no_grad():
            num_batches = (len(data_tensor) + batch_size - 1) // batch_size
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(data_tensor))

                inputs = data_tensor[start_idx:end_idx].to(self.device)
                targets = label_tensor[start_idx:end_idx].to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item() * (end_idx - start_idx)
                _, preds = torch.max(outputs, 1)
                total += targets.size(0)
                total_correct += (preds == targets).sum().item()

        avg_loss = total_loss / total if total > 0 else float("inf")
        acc = 100.0 * total_correct / total if total > 0 else 0.0
        return avg_loss, acc

    def _save_projection_data(
        self,
        variant_key: str,
        epoch_num: int,
        coords_2d: np.ndarray,
        n_attackers: int,
        defense_stats: Dict,
    ) -> str:
        path = os.path.join(self.plot_data_dir, f"{variant_key}_best_projection.npz")
        role_labels = np.array([
            "malicious" if i < n_attackers else "benign" for i in range(len(coords_2d))
        ])
        np.savez(
            path,
            coords=coords_2d,
            role_labels=role_labels,
            epoch=np.array([epoch_num]),
            tpr=np.array([defense_stats.get("detection_rate", 0.0)]),
            fpr=np.array([defense_stats.get("false_positive_rate", 0.0)]),
        )
        return path

    def run_variant(
        self,
        variant_key: str,
        user_tr_data_tensors: List[torch.Tensor],
        user_tr_label_tensors: List[torch.Tensor],
        val_data_tensor: torch.Tensor,
        val_label_tensor: torch.Tensor,
        te_data_tensor: torch.Tensor,
        te_label_tensor: torch.Tensor,
    ) -> AblationResult:
        scheme_name = self.VARIANT_MAP[variant_key][0]
        self.logger.info(f"开始运行消融方案: {scheme_name}")

        nusers = len(user_tr_data_tensors)
        n_attackers = min(self.config.num_attackers, nusers - 1)
        batch_size = self.config.batch_size
        num_local_epochs = self.config.num_local_epochs
        fed_lr = self.config.learning_rate

        fed_model, optimizer_fed = ModelFactory.create_model(self.config.dataset, self.config)
        fed_model.to(self.device)
        scheduler = StepLRScheduler(optimizer_fed, init_lr=fed_lr)
        criterion = nn.CrossEntropyLoss()

        input_shape = tuple(user_tr_data_tensors[0][0].size())
        defender = self._create_ablation_defender(variant_key, fed_model, nusers, input_shape)
        self.wavelet_defender = defender
        evaluator = DefenseEvaluator(n_attackers, nusers)
        attack_fn = AttackFactory.create_attack("fang", self.config)

        best_test_acc = -1.0
        best_val_acc = -1.0
        best_epoch = -1
        best_defense_stats = {}
        best_plot_data_path = ""

        user_tr_len = user_tr_data_tensors[0].size(0)
        nbatches = (user_tr_len + batch_size - 1) // batch_size
        client_weights = [len(data) for data in user_tr_data_tensors]

        for epoch_num in range(self.config.num_epochs):
            self.logger.info(f"[{scheme_name}] ===== 联邦轮次 {epoch_num} =====")

            all_client_models = []
            train_loss = 0.0

            for i in range(nusers):
                client_model, client_optimizer = ModelFactory.create_model(self.config.dataset, self.config)
                client_model.to(self.device)
                client_model.load_state_dict(fed_model.state_dict())
                for param_group in client_optimizer.param_groups:
                    param_group["lr"] = fed_lr

                if i >= n_attackers:
                    client_loss, _, _ = self._train_client_model(
                        client_model,
                        criterion,
                        client_optimizer,
                        user_tr_data_tensors[i],
                        user_tr_label_tensors[i],
                        nbatches,
                        batch_size,
                        user_tr_len,
                        num_local_epochs,
                        self.device,
                        i,
                    )
                    train_loss += client_loss

                all_client_models.append(client_model)

            benign_updates = []
            for i in range(n_attackers, nusers):
                params = [param.data.view(-1) for param in all_client_models[i].parameters()]
                benign_updates.append(torch.cat(params))

            if benign_updates:
                benign_updates = torch.stack(benign_updates)
                global_params = torch.cat([param.data.view(-1) for param in fed_model.parameters()])
                for i in range(n_attackers):
                    mal_update = attack_fn(benign_updates, global_params, n_attackers)
                    idx = 0
                    for param in all_client_models[i].parameters():
                        size = param.numel()
                        param.data = mal_update[idx:idx + size].reshape(param.shape).to(self.device)
                        idx += size

            all_client_params = []
            for client_model in all_client_models:
                params = [param.data.view(-1) for param in client_model.parameters()]
                all_client_params.append(torch.cat(params))
            all_client_params_tensor = torch.stack(all_client_params)

            probed_updates = defender.generate_probing_parameters(all_client_params_tensor)
            coords_2d, _ = defender.compute_feature_projection(all_client_params_tensor, probed_updates)

            true_malicious_indices = list(range(n_attackers))
            malicious_indices = defender.detect_malicious_clients(
                all_client_params_tensor, probed_updates, epoch_num, true_malicious_indices
            )
            selected_indices = [i for i in range(nusers) if i not in malicious_indices]
            defense_stats = evaluator.evaluate_round(selected_indices, malicious_indices, epoch_num)

            if selected_indices:
                self._fedavg_aggregate(
                    fed_model,
                    [all_client_models[i] for i in selected_indices],
                    [client_weights[i] for i in selected_indices],
                )

            scheduler.step(epoch_num)
            val_loss, val_acc = self._evaluate_model(fed_model, criterion, val_data_tensor, val_label_tensor, 64)
            test_loss, test_acc = self._evaluate_model(fed_model, criterion, te_data_tensor, te_label_tensor, 128)

            avg_train_loss = train_loss / max(nusers - n_attackers, 1)
            self.logger.info(
                f"[{scheme_name}] epoch={epoch_num}, train_loss={avg_train_loss:.4f}, "
                f"val_acc={val_acc:.2f}%, test_acc={test_acc:.2f}%, "
                f"TPR={defense_stats['detection_rate'] * 100:.2f}%, "
                f"FPR={defense_stats['false_positive_rate'] * 100:.2f}%"
            )

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_val_acc = val_acc
                best_epoch = epoch_num
                best_defense_stats = defense_stats
                best_plot_data_path = self._save_projection_data(
                    variant_key, epoch_num, coords_2d, n_attackers, defense_stats
                )

        return AblationResult(
            scheme=scheme_name,
            best_epoch=best_epoch,
            best_test_acc=best_test_acc,
            best_val_acc=best_val_acc,
            tpr=best_defense_stats.get("detection_rate", 0.0) * 100.0,
            fpr=best_defense_stats.get("false_positive_rate", 0.0) * 100.0,
            plot_data_path=best_plot_data_path,
        )


def run_ablation_experiments(config: FLConfig, output_root: str) -> List[AblationResult]:
    user_tr_data_tensors, user_tr_label_tensors, val_data_tensor, val_label_tensor, te_data_tensor, te_label_tensor = (
        DataManager.load_and_split_data(config)
    )

    variant_keys = ["raw_diff", "dct", "dwt"]
    results = []

    for variant_key in variant_keys:
        variant_output = os.path.join(output_root, variant_key)
        os.makedirs(variant_output, exist_ok=True)
        variant_config = copy.deepcopy(config)
        variant_config.output_dir = variant_output
        runner = SpecShieldAblationRunner(variant_config, output_root)
        result = runner.run_variant(
            variant_key,
            user_tr_data_tensors,
            user_tr_label_tensors,
            val_data_tensor,
            val_label_tensor,
            te_data_tensor,
            te_label_tensor,
        )
        results.append(result)

    metrics_df = pd.DataFrame([
        {
            "Scheme": result.scheme,
            "Best Epoch": result.best_epoch,
            "Best Test Acc. (%)": result.best_test_acc,
            "TPR (%)": result.tpr,
            "FPR (%)": result.fpr,
            "Plot Data": result.plot_data_path,
        }
        for result in results
    ])
    metrics_csv = os.path.join(output_root, "ablation_metrics.csv")
    metrics_json = os.path.join(output_root, "ablation_metrics.json")
    metrics_df.to_csv(metrics_csv, index=False)
    with open(metrics_json, "w") as f:
        json.dump(metrics_df.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

    print(f"消融实验指标已保存到: {metrics_csv}")
    print(f"绘图散点数据目录: {os.path.join(output_root, 'plot_data')}")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SpecShield 消融实验（CIFAR-10 / Fang）")
    parser.add_argument("--data-dir", default="./data", help="数据目录；tiny-ImageNet 可放在工程根目录同级")
    parser.add_argument("--output-dir", default="ablation_results_cifar10_fang", help="实验输出目录")
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--num-attackers", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--num-local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    args = build_parser().parse_args()
    config = FLConfig(
        dataset="cifar10",
        data_dir=args.data_dir,
        num_clients=args.num_clients,
        num_attackers=args.num_attackers,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        num_local_epochs=args.num_local_epochs,
        attack_type="fang",
        deviation_type="unit_vec",
        defense_enabled=True,
        seed=args.seed,
        output_dir=args.output_dir,
        use_gpu=args.use_gpu,
        is_iid=True,
        verbose_logging=False,
        data_debug_logging=False,
        model_debug_logging=False,
        save_all_snapshots=False,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    run_ablation_experiments(config, args.output_dir)


if __name__ == "__main__":
    main()
