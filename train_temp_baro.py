#!/usr/bin/env python3
"""
Step 4: Training Pipeline
==========================
Complete training for IMUNet-ResNet with:
    - Dataset loading from .npz files (8-channel slice from 15-ch .npz)
    - Two-stage training: MSE first, then Gaussian NLL with uncertainty
    - Validation with trajectory reconstruction
    - Model export to ONNX / TorchScript for Android
    - Drift analysis
    - W&B logging: all metrics, per-axis losses, per-epoch random-flight ATE

8-channel input (channels sliced from 15-ch .npz at DataLoader time):
    Ch 0-2  gyro_xyz     (rad/s)
    Ch 3-5  accel_xyz    (m/s²)
    Ch 6    baro_p       (Pa)
    Ch 7    baro_t       (°C)

Warm-start from 6-channel checkpoint:
    python train.py --dataset dataset/ --epochs 100 --stage 1 \\
        --checkpoint old_model_epoch100.pt --arch resnet50

Usage:
    # Stage 1: Train with MSE loss (no uncertainty)
    python train.py --dataset dataset/ --epochs 100 --stage 1

    # Stage 2: Fine-tune with Gaussian NLL (adds uncertainty head)
    python train.py --dataset dataset/ --epochs 50 --stage 2 --resume checkpoints/stage1_best.pt

    # Full pipeline (both stages)
    python train.py --dataset dataset/ --stage both

    # Export trained model
    python train.py --dataset dataset/ --export --resume checkpoints/stage2_best.pt
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import wandb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (
    IMUNetResNet, IMUNetResNet50,
    VelocityMSELoss, GaussianNLLLoss,
    DisplacementWeightedLoss, GaussianNLLDisplacementLoss,
)
# ← CHANGE: import stem-surgery helper for 6-ch → 8-ch warm-start
from load_checkpoint import load_pretrained_stem


# ────────────────────────────────────────────
# CONSTANTS
# ← CHANGE: explicit channel count used throughout
# ────────────────────────────────────────────

IN_CHANNELS   = 8          # channels the model sees (slice of 15-ch .npz)
CHANNEL_SLICE = slice(0, 8)


# ────────────────────────────────────────────
# DATASET
# ────────────────────────────────────────────

def _load_one_npz(path, target_key):
    """
    Load a single .npz file and return (windows, targets, n_windows).
    Runs inside a ThreadPoolExecutor thread — keep it pure numpy, no torch.

    ← CHANGE: slices channels [0:8] from the 15-ch imu_windows array.
    The remaining 7 channels (mag_xyz, quat_wxyz) are discarded here so
    every downstream consumer automatically gets the right shape.
    """
    data    = np.load(path)
    windows = data["imu_windows"][:, CHANNEL_SLICE, :].astype(np.float32)  # (N, 8, T)
    targets = data[target_key].astype(np.float32)                           # (N, 3)
    return windows, targets, len(windows)


class IMUDataset(Dataset):
    """
    Loads .npz files produced by extract_training_data.py (15-ch output).
    Serves (imu_window, velocity_gt) pairs with channels sliced to [0:8].

    Optimisations vs the naive version
    ────────────────────────────────────
    1. Parallel I/O  — files are read with a ThreadPoolExecutor so decompression
       overlaps across cores instead of being purely serial.
    2. Pre-allocated output arrays — we do one pass to collect shapes, allocate
       the full (N, 8, T) array once, then fill it in-place.
    3. Vectorised normalisation — single numpy broadcast eliminates Python loop.
    4. Cache — on first load the processed arrays are saved as a single .npz
       alongside the source data.  Subsequent loads skip all I/O.
    5. Zero-copy __getitem__ — stores data as contiguous float32 numpy array
       and uses torch.from_numpy() (zero-copy view).
    """

    # ← CHANGE: bumped from 2 → 3 because the channel slice changes the
    # normalised values stored in cache; old v2 caches are invalid.
    _CACHE_VERSION = 3

    def __init__(self, data_dir, target='velocity', normalize=True,
                 num_io_workers=8, cache=True):
        t_start = time.time()

        info_path = os.path.join(data_dir, "dataset_info.json")
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"dataset_info.json not found in {data_dir}")
        with open(info_path) as f:
            self.info = json.load(f)

        # ← CHANGE: slice [:IN_CHANNELS] from the normalization stats.
        # dataset_info.json now contains 15-ch stats (from extract_training_data.py).
        # The model only sees channels 0–7, so we take only those means/stds.
        self.means = np.array(
            self.info["normalization"]["channel_means"], dtype=np.float32
        )[:IN_CHANNELS]
        self.stds = np.array(
            self.info["normalization"]["channel_stds"], dtype=np.float32
        )[:IN_CHANNELS]
        self.stds[self.stds < 1e-8] = 1.0

        npz_files = sorted(Path(data_dir).glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files in {data_dir}")
        npz_files = [p for p in npz_files if not p.name.startswith("_cache_")]

        target_key = ("displacement_gt" if target == "displacement" else "velocity_gt")

        # ← CHANGE: cache filename now includes IN_CHANNELS so a 6-ch, 8-ch,
        # and 15-ch run in the same dataset directory never share a cache.
        cache_path = (
            Path(data_dir) /
            f"_cache_{target}_{len(npz_files)}_ch{IN_CHANNELS}_v{self._CACHE_VERSION}.npz"
        )

        loaded_from_cache = False
        if cache and cache_path.exists():
            try:
                print(f"  Loading from cache: {cache_path.name}")
                c = np.load(cache_path, allow_pickle=False)
                self.windows     = c["windows"]
                self.targets     = c["targets"]
                self.file_slices = list(zip(
                    c["slice_starts"].tolist(),
                    c["slice_ends"].tolist(),
                ))
                loaded_from_cache = True
                print(f"  Cache hit — {len(self.windows)} windows in "
                      f"{time.time() - t_start:.2f}s")
            except Exception as e:
                print(f"  Cache read failed ({e}), rebuilding…")

        if not loaded_from_cache:
            print(f"  Loading {len(npz_files)} files with {num_io_workers} workers…")

            futures = {}
            results = [None] * len(npz_files)
            with ThreadPoolExecutor(max_workers=num_io_workers) as ex:
                for i, path in enumerate(npz_files):
                    futures[ex.submit(_load_one_npz, path, target_key)] = i
                for fut in as_completed(futures):
                    idx          = futures[fut]
                    results[idx] = fut.result()

            ns      = [r[2] for r in results]
            total_n = sum(ns)
            T       = results[0][0].shape[2]
            ch      = results[0][0].shape[1]   # 8 after CHANNEL_SLICE

            self.windows = np.empty((total_n, ch, T), dtype=np.float32)
            self.targets = np.empty((total_n, 3),     dtype=np.float32)

            self.file_slices = []
            cursor = 0
            for w, t, n in results:
                self.windows[cursor:cursor + n] = w
                self.targets[cursor:cursor + n] = t
                self.file_slices.append((cursor, cursor + n))
                cursor += n

            if normalize:
                means = self.means[:, np.newaxis]   # (8, 1)
                stds  = self.stds[:, np.newaxis]    # (8, 1)
                self.windows -= means
                self.windows /= stds

            if cache:
                print(f"  Saving cache → {cache_path.name}")
                np.savez_compressed(
                    cache_path,
                    windows      = self.windows,
                    targets      = self.targets,
                    slice_starts = np.array([s for s, _ in self.file_slices]),
                    slice_ends   = np.array([e for _, e in self.file_slices]),
                )

            print(f"  Loaded {len(self.windows)} windows from {len(npz_files)} files "
                  f"in {time.time() - t_start:.2f}s")

        print(f"  Target: {target}  |  Flights: {self.n_flights}  |  "
              f"Channels: {self.windows.shape[1]}")

        self.windows = np.ascontiguousarray(self.windows, dtype=np.float32)
        self.targets = np.ascontiguousarray(self.targets, dtype=np.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.windows[idx]),
            torch.from_numpy(self.targets[idx]),
        )

    def get_flight(self, file_idx):
        s, e = self.file_slices[file_idx]
        return self.windows[s:e], self.targets[s:e]

    @property
    def n_flights(self):
        return len(self.file_slices)


# ────────────────────────────────────────────
# W&B HELPERS
# ────────────────────────────────────────────

def init_wandb(args, stage):
    run = wandb.init(
        project = "IMUNet-Odometry",
        name    = f"stage{stage}_{args.target}_8ch",
        group   = f"stage{stage}",
        config  = {
            "stage":       stage,
            "epochs":      args.epochs,
            "batch_size":  args.batch_size,
            "lr":          args.lr,
            "dropout":     args.dropout,
            "window":      args.window,
            "target":      args.target,
            "val_split":   args.val_split,
            "in_channels": IN_CHANNELS,       # ← CHANGE: log channel count
        },
        reinit = True,
    )

    wandb.define_metric("epoch")
    for prefix in ("train", "val", "optimizer", "scheduler", "drift"):
        wandb.define_metric(f"{prefix}/*", step_metric="epoch")

    return run


def _wb_loss_dict(prefix, metrics):
    return {f"{prefix}/{k}": v for k, v in metrics.items()}


# ────────────────────────────────────────────
# TRAINING
# ────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, stage):
    model.train()
    total_loss = 0
    n_batches  = 0

    for imu, target in loader:
        imu, target = imu.to(device), target.to(device)
        optimizer.zero_grad()

        if stage == 1:
            pred = model(imu)
            loss = criterion(pred, target)
        else:
            vel, logvar = model(imu)
            loss = criterion(vel, logvar, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / n_batches


def validate(model, loader, criterion, device, stage):
    model.eval()
    total_loss  = 0
    all_preds   = []
    all_targets = []
    n_batches   = 0

    with torch.no_grad():
        for imu, target in loader:
            imu, target = imu.to(device), target.to(device)

            if stage == 1:
                pred = model(imu)
                loss = criterion(pred, target)
                all_preds.append(pred.cpu().numpy())
            else:
                vel, logvar = model(imu)
                loss = criterion(vel, logvar, target)
                all_preds.append(vel.cpu().numpy())

            all_targets.append(target.cpu().numpy())
            total_loss += loss.item()
            n_batches  += 1

    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    rmse       = np.sqrt(np.mean((all_preds - all_targets) ** 2, axis=0))
    total_rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))

    pred_mag   = np.sqrt(np.sum(all_preds   ** 2, axis=1))
    target_mag = np.sqrt(np.sum(all_targets ** 2, axis=1))
    mag_rmse   = np.sqrt(np.mean((pred_mag - target_mag) ** 2))

    mae = np.mean(np.abs(all_preds - all_targets), axis=0)

    return {
        "loss":           total_loss / n_batches,
        "rmse_total":     float(total_rmse),
        "rmse_vx":        float(rmse[0]),
        "rmse_vy":        float(rmse[1]),
        "rmse_vz":        float(rmse[2]),
        "rmse_magnitude": float(mag_rmse),
        "mae_vx":         float(mae[0]),
        "mae_vy":         float(mae[1]),
        "mae_vz":         float(mae[2]),
    }


# ────────────────────────────────────────────
# PER-FLIGHT TRAJECTORY METRICS
# ─────────────────────────────────────────────────────────────────────────────
# ← CHANGE: compute_flight_metrics() now computes ATE/100m and ATE/min
# alongside the existing FDE.  FDE (final displacement error) is preserved
# unchanged — it measures "how far from the destination" for each flight.
# The two new metrics normalise mean position error so the number is
# comparable across flights of different lengths:
#
#   ATE/100m = mean_pos_error / (total_true_dist_m / 100)
#              "metres of position error per 100 metres of true flight"
#
#   ATE/min  = mean_pos_error / (total_true_dur_s  / 60)
#              "metres of position error per minute of true flight"
#
# Both denominators are printed in the console line so the raw number
# is always interpretable even without W&B open.
# ────────────────────────────────────────────

def _infer_flight(model, windows, device, stage, batch_size=512):
    preds = []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = torch.from_numpy(
                windows[i:i + batch_size]
            ).to(device, non_blocking=True)
            pred = model(batch)
            if stage == 2:
                pred = pred[0]
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds)


def compute_flight_metrics(model, dataset, device, stage,
                           dt=0.1, n_flights=10, rng=None):
    """
    Sample n_flights random flights and compute:
      - FDE       : Final Displacement Error [m]  (existing metric, unchanged)
      - ATE/100m  : mean position error per 100 m of true flight [m/100m]
      - ATE/min   : mean position error per minute of true flight [m/min]

    Returns dict with aggregates over sampled flights, or None if no flights.
    """
    if rng is None:
        rng = np.random.default_rng()

    available = dataset.n_flights
    if available == 0:
        return None

    k      = min(n_flights, available)
    chosen = rng.choice(available, size=k, replace=False).tolist()

    model.eval()

    fdes        = []
    ate_100ms   = []   # ATE/100m per flight
    ate_mins    = []   # ATE/min  per flight
    dist_ms     = []   # true flight distance [m]
    dur_ss      = []   # true flight duration [s]
    indices     = []
    lengths     = []

    for idx in chosen:
        windows, targets = dataset.get_flight(idx)
        if len(windows) < 2:
            continue

        preds    = _infer_flight(model, windows, device, stage)
        pred_pos = np.cumsum(preds   * dt, axis=0)   # (N, 3)
        gt_pos   = np.cumsum(targets * dt, axis=0)   # (N, 3)

        # FDE — unchanged
        fde = float(np.linalg.norm(pred_pos[-1] - gt_pos[-1]))

        # ← CHANGE: per-step position error → mean ATE for this flight
        pos_errors   = np.linalg.norm(pred_pos - gt_pos, axis=1)   # (N,)
        mean_pos_err = float(pos_errors.mean())

        # True flight distance and duration
        step_dists = np.linalg.norm(targets * dt, axis=1)   # (N,)
        total_dist = float(step_dists.sum())                 # metres
        total_dur  = len(targets) * dt                       # seconds

        # Normalised ATE (guard against zero-distance flights like hover)
        ate_100m = (mean_pos_err / max(total_dist, 1.0)) * 100.0
        ate_min  = (mean_pos_err / max(total_dur,  1.0)) * 60.0

        fdes.append(fde)
        ate_100ms.append(ate_100m)
        ate_mins.append(ate_min)
        dist_ms.append(total_dist)
        dur_ss.append(total_dur)
        indices.append(idx)
        lengths.append(len(windows))

    if not fdes:
        return None

    fdes_arr     = np.array(fdes)
    ate_100m_arr = np.array(ate_100ms)
    ate_min_arr  = np.array(ate_mins)

    # ← CHANGE: expanded W&B table with new columns
    table = wandb.Table(
        columns=["flight_idx", "flight_len_windows",
                 "fde_m", "ate_per_100m", "ate_per_min",
                 "true_dist_m", "true_dur_s"],
        data=[
            [indices[i], lengths[i], fdes[i],
             ate_100ms[i], ate_mins[i],
             dist_ms[i], dur_ss[i]]
            for i in range(len(fdes))
        ],
    )

    return {
        # FDE (unchanged)
        "fde_mean":       float(fdes_arr.mean()),
        "fde_min":        float(fdes_arr.min()),
        "fde_max":        float(fdes_arr.max()),
        "fde_std":        float(fdes_arr.std()),
        # ← CHANGE: ATE normalised metrics
        "ate_per_100m_mean": float(ate_100m_arr.mean()),
        "ate_per_100m_std":  float(ate_100m_arr.std()),
        "ate_per_min_mean":  float(ate_min_arr.mean()),
        "ate_per_min_std":   float(ate_min_arr.std()),
        # Context for console print
        "total_dist_m":   float(sum(dist_ms)),
        "total_dur_s":    float(sum(dur_ss)),
        "n_evaluated":    len(fdes),
        "table":          table,
    }


# ────────────────────────────────────────────
# MODEL FACTORY
# ← CHANGE: in_channels is now a parameter (was hardcoded 6)
# ────────────────────────────────────────────

def build_model(arch, output_dim, window_size, dropout, in_channels=IN_CHANNELS):
    if arch == "resnet50":
        return IMUNetResNet50(
            in_channels=in_channels, window_size=window_size,
            output_dim=output_dim, dropout=dropout,
        )
    return IMUNetResNet(
        in_channels=in_channels, window_size=window_size,
        output_dim=output_dim, dropout=dropout,
    )


# ────────────────────────────────────────────
# MAIN TRAINING LOOP
# ────────────────────────────────────────────

def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion,
                device, epochs, stage, save_dir,
                dataset_full=None, dt=0.1, n_eval_flights=10):
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")
    history       = []
    rng           = np.random.default_rng(seed=0)

    for epoch in range(1, epochs + 1):
        t0          = time.time()
        train_loss  = train_epoch(model, train_loader, optimizer, criterion, device, stage)
        val_metrics = validate(model, val_loader, criterion, device, stage)
        dt_epoch    = time.time() - t0

        if scheduler:
            scheduler.step(val_metrics["loss"])

        lr = optimizer.param_groups[0]["lr"]

        fm = None
        if dataset_full is not None and dataset_full.n_flights > 0:
            fm = compute_flight_metrics(
                model, dataset_full, device, stage,
                dt=dt, n_flights=n_eval_flights, rng=rng,
            )

        # ── W&B log ───────────────────────────────────────────────────────────
        wb_log = {
            "epoch":          epoch,
            "train/loss":     train_loss,
            "optimizer/lr":   lr,
            "time/epoch_sec": dt_epoch,
        }
        wb_log.update(_wb_loss_dict("val", val_metrics))

        if fm is not None:
            wb_log["val/FDE_mean"]          = fm["fde_mean"]
            wb_log["val/FDE_min"]           = fm["fde_min"]
            wb_log["val/FDE_max"]           = fm["fde_max"]
            wb_log["val/FDE_std"]           = fm["fde_std"]
            wb_log["val/FDE_n_flights"]     = fm["n_evaluated"]
            wb_log["val/FDE_table"]         = fm["table"]
            # ← CHANGE: ATE normalised metrics in W&B
            wb_log["val/ATE_per_100m_mean"] = fm["ate_per_100m_mean"]
            wb_log["val/ATE_per_100m_std"]  = fm["ate_per_100m_std"]
            wb_log["val/ATE_per_min_mean"]  = fm["ate_per_min_mean"]
            wb_log["val/ATE_per_min_std"]   = fm["ate_per_min_std"]

        wandb.log(wb_log)

        # ── History ───────────────────────────────────────────────────────────
        history.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr":         lr,
            **({"fde_mean":        fm["fde_mean"],
                "fde_std":         fm["fde_std"],
                "ate_per_100m":    fm["ate_per_100m_mean"],  # ← CHANGE
                "ate_per_min":     fm["ate_per_min_mean"],   # ← CHANGE
            } if fm else {}),
        })

        # ── Console print ─────────────────────────────────────────────────────
        improved = ""
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":             best_val_loss,
                "stage":                stage,
                "in_channels":          IN_CHANNELS,   # ← CHANGE: save for eval reload
            }, os.path.join(save_dir, f"stage{stage}_best.pt"))
            wandb.summary[f"stage{stage}_best_val_loss"] = best_val_loss
            wandb.summary[f"stage{stage}_best_epoch"]    = epoch
            improved = " ★"

        # ← CHANGE: traj_str now includes ATE/100m, ATE/min, and context
        # so the number is always interpretable from the console alone.
        if fm is not None:
            dist_km  = fm["total_dist_m"] / 1000.0
            dur_min  = fm["total_dur_s"]  / 60.0
            traj_str = (
                f"  FDE={fm['fde_mean']:.3f}±{fm['fde_std']:.3f}m"
                f"  ATE/100m={fm['ate_per_100m_mean']:.4f}m"
                f"  ATE/min={fm['ate_per_min_mean']:.4f}m"
                f"  [{fm['n_evaluated']} flights, "
                f"≈{fm['total_dist_m']:.0f}m / {dur_min:.1f}min total]"
            )
        else:
            traj_str = ""

        print(
            f"  Epoch {epoch:3d}/{epochs}  "
            f"train={train_loss:.5f}  val={val_metrics['loss']:.5f}  "
            f"rmse={val_metrics['rmse_total']:.4f}  "
            f"vx={val_metrics['rmse_vx']:.4f} "
            f"vy={val_metrics['rmse_vy']:.4f} "
            f"vz={val_metrics['rmse_vz']:.4f}  "
            f"lr={lr:.1e}  {dt_epoch:.1f}s"
            f"{traj_str}{improved}"
        )

        if epoch % 20 == 0:
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "stage":                stage,
                "in_channels":          IN_CHANNELS,
            }, os.path.join(save_dir, f"stage{stage}_epoch{epoch}.pt"))

    with open(os.path.join(save_dir, f"stage{stage}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return history


# ────────────────────────────────────────────
# DRIFT ANALYSIS  (unchanged — operates on integrated trajectories)
# ────────────────────────────────────────────

def analyze_drift(model, dataset, device, stage, dt=0.005):
    """
    Per-flight drift analysis with fully signed per-axis statistics.
    (Unchanged from original — see original docstring for full explanation.)
    """
    model.eval()
    intervals       = [10, 30, 60, 120, 300]
    samples_per_sec = 1.0 / dt

    signed_errs    = [[] for _ in intervals]
    n_flights_used = 0

    with torch.no_grad():
        for fi in range(dataset.n_flights):
            windows, targets = dataset.get_flight(fi)
            if len(windows) < 2:
                continue

            all_preds = []
            for i in range(0, len(windows), 512):
                batch = torch.from_numpy(
                    windows[i:i + 512]
                ).to(device, non_blocking=True)
                if stage == 1:
                    pred = model(batch)
                else:
                    pred, _ = model(batch)
                all_preds.append(pred.cpu().numpy())

            preds    = np.concatenate(all_preds)
            pred_pos = np.cumsum(preds   * dt, axis=0)
            gt_pos   = np.cumsum(targets * dt, axis=0)
            err      = pred_pos - gt_pos

            for i, t in enumerate(intervals):
                idx = int(t * samples_per_sec) - 1
                if idx < len(err):
                    signed_errs[i].append(err[idx])

            n_flights_used += 1

    if n_flights_used == 0:
        print("  Drift analysis: no usable flights found.")
        return {}

    print(f"\n  Drift Analysis — stage {stage}  ({n_flights_used} flights, dt={dt}s)")
    print(f"\n  {'Time':>6}  "
          f"{'ex mean':>9} {'ey mean':>9} {'ez mean':>9}  "
          f"{'ex std':>8} {'ey std':>8} {'ez std':>8}  "
          f"{'ex rms':>8} {'ey rms':>8} {'ez rms':>8}")
    print(f"  {'-'*105}")

    table_rows = []

    for i, t in enumerate(intervals):
        if not signed_errs[i]:
            continue

        s      = np.stack(signed_errs[i], axis=0)
        s_mean = s.mean(axis=0)
        s_std  = s.std(axis=0)
        s_rms  = np.sqrt((s ** 2).mean(axis=0))

        print(
            f"  {t:5d}s  "
            f"{s_mean[0]:+9.3f} {s_mean[1]:+9.3f} {s_mean[2]:+9.3f}m  "
            f"{s_std[0]:8.3f} {s_std[1]:8.3f} {s_std[2]:8.3f}m  "
            f"{s_rms[0]:8.3f} {s_rms[1]:8.3f} {s_rms[2]:8.3f}m"
        )

        table_rows.append([
            t,
            float(s_mean[0]), float(s_mean[1]), float(s_mean[2]),
            float(s_std[0]),  float(s_std[1]),  float(s_std[2]),
            float(s_rms[0]),  float(s_rms[1]),  float(s_rms[2]),
        ])

    print(f"\n  Legend:")
    print(f"    mean  — signed bias per axis  (near 0 = errors cancel = unbiased)")
    print(f"    std   — flight-to-flight spread of signed errors")
    print(f"    rms   — sqrt(bias²+spread²), overall accuracy per axis")

    if table_rows:
        tbl = wandb.Table(
            columns=[
                "time_s",
                "ex_mean_m", "ey_mean_m", "ez_mean_m",
                "ex_std_m",  "ey_std_m",  "ez_std_m",
                "ex_rms_m",  "ey_rms_m",  "ez_rms_m",
            ],
            data=table_rows,
        )
        wandb.log({f"drift/stage{stage}": tbl})

        last = table_rows[-1]
        t_last = last[0]
        wandb.summary.update({
            f"drift/stage{stage}_{t_last}s_ex_mean_m": last[1],
            f"drift/stage{stage}_{t_last}s_ey_mean_m": last[2],
            f"drift/stage{stage}_{t_last}s_ez_mean_m": last[3],
            f"drift/stage{stage}_{t_last}s_ex_rms_m":  last[7],
            f"drift/stage{stage}_{t_last}s_ey_rms_m":  last[8],
            f"drift/stage{stage}_{t_last}s_ez_rms_m":  last[9],
        })

    return {
        "n_flights":   n_flights_used,
        "intervals":   intervals,
        "signed_errs": signed_errs,
    }


# ────────────────────────────────────────────
# BEST-MODEL EVALUATION
# ← CHANGE: uses arch arg instead of hardcoded IMUNetResNet (ResNet18).
# Previously evaluate_best_model always loaded ResNet18 regardless of
# what was actually trained — silently wrong for ResNet50 checkpoints.
# ────────────────────────────────────────────

def evaluate_best_model(checkpoint_path, dataset, device,
                        arch="resnet50",
                        window_size=200, dropout=0.0,
                        dt=0.005, intervals=None):
    """
    Load the best saved checkpoint, run it over every flight in `dataset`,
    and report bias_magnitude (A) and mean_pos_error (B) at fixed time
    checkpoints.  See original docstring for full metric explanation.

    ← CHANGE: `arch` parameter added (was hardcoded to resnet18).
    """
    if intervals is None:
        intervals = [10, 30, 60, 120, 300]

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt    = torch.load(checkpoint_path, map_location=device)
    stage   = ckpt.get("stage", 1)
    out_dim = 6 if stage == 2 else 3

    # ← CHANGE: respects arch flag
    model = build_model(
        arch        = arch,
        output_dim  = out_dim,
        window_size = window_size,
        dropout     = dropout,
        in_channels = IN_CHANNELS,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"\n  Loaded: {checkpoint_path}")
    print(f"  Arch: {arch}  |  Stage {stage}  |  output_dim={out_dim}  |  "
          f"{model.get_num_params():,} params")

    samples_per_sec = 1.0 / dt
    signed_errs     = [[] for _ in intervals]
    n_flights_used  = 0

    with torch.no_grad():
        for fi in range(dataset.n_flights):
            windows, targets = dataset.get_flight(fi)
            if len(windows) < 2:
                continue

            all_preds = []
            for i in range(0, len(windows), 512):
                batch = torch.from_numpy(
                    windows[i:i + 512]
                ).to(device, non_blocking=True)
                pred = model(batch)
                if out_dim == 6:
                    pred = pred[0]
                all_preds.append(pred.cpu().numpy())

            preds    = np.concatenate(all_preds)
            pred_pos = np.cumsum(preds   * dt, axis=0)
            gt_pos   = np.cumsum(targets * dt, axis=0)
            err      = pred_pos - gt_pos

            for i, t in enumerate(intervals):
                idx = int(t * samples_per_sec) - 1
                if idx < len(err):
                    signed_errs[i].append(err[idx])

            n_flights_used += 1

    print(f"  Evaluated {n_flights_used} flights\n")

    print(f"  {'Time':>6}  "
          f"{'A: bias_mag':>12}  "
          f"{'B: mean_err':>12}  "
          f"{'bias direction (ex, ey, ez)':>30}")
    print(f"  {'-'*80}")

    results = {}
    bias_magnitude = []
    mean_pos_error = []

    for i, t in enumerate(intervals):
        if not signed_errs[i]:
            bias_magnitude.append(None)
            mean_pos_error.append(None)
            continue

        s = np.stack(signed_errs[i], axis=0)

        mean_err_vec     = s.mean(axis=0)
        A = float(np.linalg.norm(mean_err_vec))

        per_flight_norms = np.linalg.norm(s, axis=1)
        B = float(per_flight_norms.mean())

        bias_magnitude.append(A)
        mean_pos_error.append(B)

        print(
            f"  {t:5d}s  "
            f"  {A:10.4f}m  "
            f"  {B:10.4f}m  "
            f"  ({mean_err_vec[0]:+.3f}, {mean_err_vec[1]:+.3f}, {mean_err_vec[2]:+.3f})"
        )
        results[t] = {
            "bias_magnitude": A,
            "mean_pos_error": B,
            "bias_vector":    mean_err_vec.tolist(),
            "n_flights":      len(s),
        }

    print(f"\n  Legend:")
    print(f"    A  bias_mag  = norm(mean(err))  — systematic directional bias")
    print(f"                   ≈ 0 means errors cancel across flights (unbiased)")
    print(f"    B  mean_err  = mean(norm(err))  — average position error (ATE-style)")
    print(f"                   always ≥ 0, does not cancel, use this for accuracy")

    return {
        "intervals":      intervals,
        "bias_magnitude": bias_magnitude,
        "mean_pos_error": mean_pos_error,
        "signed_errs":    signed_errs,
        "per_interval":   results,
        "n_flights":      n_flights_used,
    }


# ────────────────────────────────────────────
# EXPORT  (unchanged)
# ────────────────────────────────────────────

def export_onnx(model, save_path, window_size=200, in_channels=IN_CHANNELS):
    model.eval()
    dummy = torch.randn(1, in_channels, window_size)
    torch.onnx.export(
        model, dummy, save_path,
        input_names   = ["imu_input"],
        output_names  = ["velocity"],
        dynamic_axes  = {"imu_input": {0: "batch"}, "velocity": {0: "batch"}},
        opset_version = 12,
    )
    print(f"  ONNX exported: {save_path}")


def export_torchscript(model, save_path, window_size=200, in_channels=IN_CHANNELS):
    model.eval()
    dummy  = torch.randn(1, in_channels, window_size)
    traced = torch.jit.trace(model, dummy)
    traced.save(save_path)
    print(f"  TorchScript exported: {save_path}")


# ────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train IMUNet-ResNet (8-channel)")
    parser.add_argument("--dataset",    required=True,  help="Dataset directory")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--stage",      default="both", choices=["1", "2", "both"])
    parser.add_argument("--arch",       default="resnet50",
                        choices=["resnet18", "resnet50"])
    parser.add_argument("--loss",       default="displacement",
                        choices=["mse", "displacement"])
    parser.add_argument("--loss-alpha", type=float, default=1.0)
    parser.add_argument("--loss-beta",  type=float, default=1.0)
    parser.add_argument("--resume",     default=None, help="Checkpoint to resume from")
    parser.add_argument("--save-dir",   default="checkpoints")
    parser.add_argument("--val-split",  type=float, default=0.15)
    parser.add_argument("--target",     default="velocity",
                        choices=["velocity", "displacement"])
    parser.add_argument("--export",     action="store_true")
    parser.add_argument("--window",     type=int,   default=200)
    parser.add_argument("--dropout",    type=float, default=0.25)
    parser.add_argument("--dt",         type=float, default=0.1)
    parser.add_argument("--wandb-project", default="IMUNet-Odometry")
    parser.add_argument("--io-workers",    type=int, default=8)
    parser.add_argument("--no-cache",      action="store_true")
    parser.add_argument("--num-workers",   type=int, default=4)
    parser.add_argument("--eval-flights",  type=int, default=10)
    # ← CHANGE: new arg — path to old 6-ch checkpoint for stem warm-start
    parser.add_argument("--checkpoint",    default=None,
                        help="Path to old 6-channel .pt checkpoint for warm-start. "
                             "Stem Conv1d ch 0-5 are copied; ch 6-7 (baro) are "
                             "Kaiming-initialised. All other layers loaded as-is.")
    # ← CHANGE: explicit in_channels arg (auto-set to IN_CHANNELS=8, exposed
    # for documentation / future experiments)
    parser.add_argument("--in-channels",   type=int, default=IN_CHANNELS,
                        help=f"Input channels to model (default {IN_CHANNELS})")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  in_channels={args.in_channels}")

    print(f"\nLoading dataset from {args.dataset}...")
    dataset = IMUDataset(
        args.dataset,
        target         = args.target,
        num_io_workers = args.io_workers,
        cache          = not args.no_cache,
    )

    n_val   = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    _dl_kwargs = dict(
        num_workers        = args.num_workers,
        pin_memory         = True,
        persistent_workers = args.num_workers > 0,
        prefetch_factor    = 2 if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, **_dl_kwargs)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size,
                              shuffle=False, **_dl_kwargs)

    print(f"  Train: {n_train}, Val: {n_val}")
    print(f"  Flights in dataset: {dataset.n_flights}")

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    if args.stage in ("1", "both"):
        print(f"\n{'='*55}")
        print(f"  STAGE 1  |  arch={args.arch}  loss={args.loss}  "
              f"in_channels={args.in_channels}")
        print(f"{'='*55}")

        init_wandb(args, stage=1)

        # ← CHANGE: pass in_channels to build_model (was hardcoded 6)
        model = build_model(
            args.arch, output_dim=3,
            window_size=args.window, dropout=args.dropout,
            in_channels=args.in_channels,
        ).to(device)
        print(f"  Params: {model.get_num_params():,}")
        wandb.summary["n_params"] = model.get_num_params()
        wandb.summary["arch"]     = args.arch

        # ← CHANGE: warm-start from 6-ch checkpoint if provided
        if args.checkpoint:
            print(f"\n  Warm-starting from 6-ch checkpoint: {args.checkpoint}")
            load_pretrained_stem(
                model,
                args.checkpoint,
                old_in_channels = 6,
                new_in_channels = args.in_channels,
                device          = str(device),
                verbose         = True,
            )
        elif args.resume and args.stage == "1":
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"  Resumed from {args.resume}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
        if args.loss == "displacement":
            criterion = DisplacementWeightedLoss(
                alpha=args.loss_alpha, beta=args.loss_beta, dt=args.dt
            )
        else:
            criterion = VelocityMSELoss()
        print(f"  Criterion: {criterion}")

        train_model(
            model, train_loader, val_loader, optimizer, scheduler,
            criterion, device, args.epochs, stage=1,
            save_dir=args.save_dir,
            dataset_full=dataset,
            dt=args.dt,
            n_eval_flights=args.eval_flights,
        )

        analyze_drift(model, dataset, device, stage=1, dt=args.dt)
        wandb.finish()

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    if args.stage in ("2", "both"):
        print(f"\n{'='*55}")
        print(f"  STAGE 2  |  arch={args.arch}  Gaussian NLL + displacement  "
              f"in_channels={args.in_channels}")
        print(f"{'='*55}")

        init_wandb(args, stage=2)

        # ← CHANGE: pass in_channels to build_model
        model_unc = build_model(
            args.arch, output_dim=6,
            window_size=args.window, dropout=args.dropout,
            in_channels=args.in_channels,
        ).to(device)
        wandb.summary["n_params"] = model_unc.get_num_params()
        wandb.summary["arch"]     = args.arch

        stage1_path = args.resume or os.path.join(args.save_dir, "stage1_best.pt")
        if os.path.exists(stage1_path):
            ckpt       = torch.load(stage1_path, map_location=device)
            state      = ckpt["model_state_dict"]
            model_dict = model_unc.state_dict()
            pretrained = {k: v for k, v in state.items()
                          if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained)
            model_unc.load_state_dict(model_dict, strict=False)
            n_loaded = len(pretrained)
            print(f"  Loaded {n_loaded}/{len(model_dict)} layers from stage 1")
            wandb.summary["stage1_layers_loaded"] = n_loaded
        else:
            print(f"  WARN: No stage 1 checkpoint found, training from scratch")

        stage2_epochs = args.epochs // 2 if args.stage == "both" else args.epochs
        optimizer = torch.optim.AdamW(
            model_unc.parameters(), lr=args.lr * 0.1, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=8
        )
        if args.loss == "displacement":
            criterion = GaussianNLLDisplacementLoss(
                alpha=args.loss_alpha, beta=args.loss_beta, dt=args.dt
            )
        else:
            criterion = GaussianNLLLoss()
        print(f"  Criterion: {criterion}")

        train_model(
            model_unc, train_loader, val_loader, optimizer, scheduler,
            criterion, device, stage2_epochs, stage=2,
            save_dir=args.save_dir,
            dataset_full=dataset,
            dt=args.dt,
            n_eval_flights=args.eval_flights,
        )

        analyze_drift(model_unc, dataset, device, stage=2, dt=args.dt)
        wandb.finish()

    # ── Export ────────────────────────────────────────────────────────────────
    if args.export:
        print(f"\n{'='*55}")
        print(f"  EXPORTING MODEL")
        print(f"{'='*55}")

        export_dir = os.path.join(args.save_dir, "export")
        os.makedirs(export_dir, exist_ok=True)

        best_path = os.path.join(args.save_dir, "stage2_best.pt")
        if not os.path.exists(best_path):
            best_path = os.path.join(args.save_dir, "stage1_best.pt")

        if os.path.exists(best_path):
            ckpt         = torch.load(best_path, map_location="cpu")
            stage        = ckpt.get("stage", 1)
            out_dim      = 6 if stage == 2 else 3

            # ← CHANGE: use args.arch and IN_CHANNELS for export model
            model_export = build_model(
                args.arch, output_dim=out_dim,
                window_size=args.window, dropout=0.0,
                in_channels=args.in_channels,
            )
            model_export.load_state_dict(ckpt["model_state_dict"])
            model_export.eval()

            onnx_path = os.path.join(export_dir, "imunet_resnet.onnx")
            ts_path   = os.path.join(export_dir, "imunet_resnet.pt")

            export_onnx(model_export, onnx_path, args.window, args.in_channels)
            export_torchscript(model_export, ts_path, args.window, args.in_channels)

            norm_info = {
                "channel_means":  dataset.means.tolist(),
                "channel_stds":   dataset.stds.tolist(),
                "window_size":    args.window,
                "sample_rate_hz": 200,
                "output_dim":     out_dim,
                "in_channels":    args.in_channels,
                # ← CHANGE: correct channel order for 8-ch model
                "channel_order":  ["gyro_x", "gyro_y", "gyro_z",
                                   "acc_x",  "acc_y",  "acc_z",
                                   "baro_p", "baro_t"],
            }
            norm_path = os.path.join(export_dir, "inference_params.json")
            with open(norm_path, "w") as f:
                json.dump(norm_info, f, indent=2)
            print(f"  Inference params: {norm_path}")
        else:
            print(f"  ERROR: No checkpoint found to export")

    # ── Best-model evaluation ─────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  BEST-MODEL EVALUATION")
    print(f"{'='*55}")

    best_path = os.path.join(args.save_dir, "stage2_best.pt")
    if not os.path.exists(best_path):
        best_path = os.path.join(args.save_dir, "stage1_best.pt")

    if os.path.exists(best_path):
        # ← CHANGE: pass args.arch so correct architecture is loaded
        evaluate_best_model(
            checkpoint_path = best_path,
            dataset         = dataset,
            device          = device,
            arch            = args.arch,
            window_size     = args.window,
            dt              = args.dt,
        )
    else:
        print(f"  No checkpoint found at {best_path}, skipping.")

    print("\nDone!")


if __name__ == "__main__":
    main()
