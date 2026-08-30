import sys; sys.stdout.reconfigure(line_buffering=True)
import logging
logging.basicConfig(level=logging.INFO)

import os
import time
import pickle
import csv
import numpy as np
import flwr as fl
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

from model.architecture import build_model
from federated.robust_strategy import TrackedFedAvg
from federated.client import make_client_fn

# ── Fixed Config & Sharding Helper ────────────────────────────────────
NUM_CLIENTS = 5
NUM_ROUNDS = 10
WINDOW_SIZE = 20

def partition_data(X, y, n=5):
    idx = np.array_split(np.arange(len(X)), n)
    return [(X[i], y[i]) for i in idx]

def run():
    print("📂 Loading sequence data...")

    # Load both datasets
    X_cic  = np.load("data/processed/X_seq_cic.npy")
    y_cic  = np.load("data/processed/y_seq_cic.npy")
    X_edge = np.load("data/processed/X_seq_edge.npy")
    y_edge = np.load("data/processed/y_seq_edge.npy")

    with open("data/processed/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    num_classes = len(le.classes_)
    print(f"   Classes ({num_classes}): {list(le.classes_)}")

    # ── Combine both datasets ─────────────────────────────────────
    X_all = np.concatenate([X_cic, X_edge], axis=0)
    y_all = np.concatenate([y_cic, y_edge], axis=0)
    print(f"   Combined: {X_all.shape}")

    # ── Remove classes with fewer than 10 samples ─────────────────
    print("\n📊 Class distribution:")
    keep_mask = np.zeros(len(y_all), dtype=bool)
    for cls_idx in range(num_classes):
        count = np.sum(y_all == cls_idx)
        cls_name = le.classes_[cls_idx]
        print(f"   {cls_name:20s}: {count} samples",
              "⚠️ REMOVED" if count < 10 else "✅")
        if count >= 10:
            keep_mask |= (y_all == cls_idx)

    X_all = X_all[keep_mask]
    y_all = y_all[keep_mask]

    # Re-encode labels to be continuous after removal
    unique_classes = np.unique(y_all)
    remap = {old: new for new, old in enumerate(unique_classes)}
    y_all = np.array([remap[y] for y in y_all])
    num_classes = len(unique_classes)
    kept_class_names = le.classes_[unique_classes]
    print(f"\n   Kept {num_classes} classes: {list(kept_class_names)}")
    print(f"   Final dataset: {X_all.shape}")

    # ── Train/test split (no stratify to avoid issues) ────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.15, random_state=42
    )
    print(f"   Train: {X_train.shape} | Test: {X_test.shape}")

    # Save test set + class info for evaluation
    os.makedirs("saved_models", exist_ok=True)
    np.save("saved_models/X_test.npy", X_test)
    np.save("saved_models/y_test.npy", y_test)
    np.save("saved_models/class_names.npy", kept_class_names)
    print("   ✅ Test set saved")

    # ── Partition into 5 client shards ────────────────────────────
    print(f"\n📊 Partitioning across {NUM_CLIENTS} clients...")
    shards = partition_data(X_train, y_train, NUM_CLIENTS)

    # ── FedAvg strategy (TRACKED so we can recover real weights + timing) ──
    strategy = TrackedFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=lambda rnd: {
            "local_epochs": 3,
            "batch_size":   64,
            "round":        rnd,
        },
    )

    # ── Start simulation ──────────────────────────────────────────
    print(f"\n🚀 Starting Federated Training ({NUM_ROUNDS} rounds)...")
    history = fl.simulation.start_simulation(
        client_fn=make_client_fn(shards, num_classes),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 0.0},
    )

    # ── Recover the REAL trained weights ────────────────────────────
    print("\n💾 Saving final model...")
    n_features = X_all.shape[2]
    final_ndarrays = fl.common.parameters_to_ndarrays(strategy.latest_parameters)

    final_model = build_model(WINDOW_SIZE, n_features, num_classes)
    final_model.set_weights(final_ndarrays)
    final_model.save("saved_models/fl_model.h5")
    print("✅ Model saved → saved_models/fl_model.h5 (real trained weights)")

    # ── Real evaluation on the held-out test set ─────────────────────
    print("\n📊 Evaluating on test set...")
    y_pred = np.argmax(final_model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"   Test Accuracy: {acc:.4f}")
    print(f"   Macro F1:      {macro_f1:.4f}")

    # ── Overhead metrics (for the paper's Section III system-overhead table) ──
    model_size_mb = sum(w.nbytes for w in final_ndarrays) / (1024 ** 2)
    comm_cost_per_round_mb = model_size_mb * NUM_CLIENTS * 2  # upload + download
    total_comm_cost_mb = comm_cost_per_round_mb * NUM_ROUNDS
    avg_round_time_s = sum(strategy.round_times) / len(strategy.round_times)

    print(f"\n📊 Overhead metrics:")
    print(f"   Model size: {model_size_mb:.2f} MB")
    print(f"   Comm. cost/round: {comm_cost_per_round_mb:.2f} MB")
    print(f"   Total comm. cost ({NUM_ROUNDS} rounds): {total_comm_cost_mb:.2f} MB")
    print(f"   Avg. round time: {avg_round_time_s:.2f} s")
    print(f"   Peak memory: {strategy.peak_mem_mb:.2f} MB")

    # ── Save everything to CSV so it's easy to pull into the paper ─────
    with open("saved_models/federated_run_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["accuracy", round(acc * 100, 2)])
        writer.writerow(["macro_f1", round(macro_f1 * 100, 2)])
        writer.writerow(["model_size_mb", round(model_size_mb, 2)])
        writer.writerow(["comm_cost_per_round_mb", round(comm_cost_per_round_mb, 2)])
        writer.writerow(["total_comm_cost_mb", round(total_comm_cost_mb, 2)])
        writer.writerow(["avg_round_time_s", round(avg_round_time_s, 2)])
        writer.writerow(["peak_mem_mb", round(strategy.peak_mem_mb, 2)])
    print("✅ Results saved → saved_models/federated_run_results.csv")

    # ── Print training history ────────────────────────────────────
    print("\n📈 Training History:")
    losses = history.losses_distributed
    for rnd, loss in losses:
        print(f"   Round {rnd:2d} → loss: {loss:.4f}")

    return history

if __name__ == "__main__":
    run()