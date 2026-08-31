import sys; sys.stdout.reconfigure(line_buffering=True)
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'

import time
import pickle
import csv
import numpy as np
import psutil
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

from model.architecture import build_model
from federated.client import EnergyIoTClient

# ── Fixed Config ────────────────────────────────────────────────────
NUM_CLIENTS = 5
NUM_ROUNDS = 10
WINDOW_SIZE = 20
LOCAL_EPOCHS = 3
BATCH_SIZE = 64


def partition_data(X, y, n=5):
    idx = np.array_split(np.arange(len(X)), n)
    return [(X[i], y[i]) for i in idx]


def average_weights(weights_list, num_samples_list):
    """Standard FedAvg: weighted mean by number of local samples (Eq. 2)."""
    total = sum(num_samples_list)
    n_layers = len(weights_list[0])
    averaged = []
    for layer_idx in range(n_layers):
        weighted_sum = sum(
            w[layer_idx] * (n / total)
            for w, n in zip(weights_list, num_samples_list)
        )
        averaged.append(weighted_sum)
    return averaged


def run():
    print("📂 Loading sequence data...")

    X_cic  = np.load("data/processed/X_seq_cic.npy")
    y_cic  = np.load("data/processed/y_seq_cic.npy")
    X_edge = np.load("data/processed/X_seq_edge.npy")
    y_edge = np.load("data/processed/y_seq_edge.npy")

    with open("data/processed/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    num_classes = len(le.classes_)
    print(f"   Classes ({num_classes}): {list(le.classes_)}")

    X_all = np.concatenate([X_cic, X_edge], axis=0)
    y_all = np.concatenate([y_cic, y_edge], axis=0)
    print(f"   Combined: {X_all.shape}")

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

    unique_classes = np.unique(y_all)
    remap = {old: new for new, old in enumerate(unique_classes)}
    y_all = np.array([remap[y] for y in y_all])
    num_classes = len(unique_classes)
    kept_class_names = le.classes_[unique_classes]
    print(f"\n   Kept {num_classes} classes: {list(kept_class_names)}")
    print(f"   Final dataset: {X_all.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.15, random_state=42
    )
    print(f"   Train: {X_train.shape} | Test: {X_test.shape}")

    os.makedirs("saved_models", exist_ok=True)
    np.save("saved_models/X_test.npy", X_test)
    np.save("saved_models/y_test.npy", y_test)
    np.save("saved_models/class_names.npy", kept_class_names)
    print("   ✅ Test set saved")

    print(f"\n📊 Partitioning across {NUM_CLIENTS} clients...")
    shards = partition_data(X_train, y_train, NUM_CLIENTS)

    n_features = X_all.shape[2]

    # ── Build 5 client objects once (each holds its own local Keras model) ──
    print("\n🏗️  Building client models...")
    clients = []
    for cid, (X, y) in enumerate(shards):
        clients.append(EnergyIoTClient(cid, X, y, num_classes))
        print(f"   Client {cid}: {len(X)} local samples")

    # ── Global model starts from client 0's random init (matches Flower's
    #    "request initial parameters from one random client" behavior) ──
    global_weights = clients[0].get_parameters(config={})

    round_times = []
    peak_mem_mb = 0.0

    print(f"\n🚀 Starting Federated Training ({NUM_ROUNDS} rounds, sequential, no Ray)...")
    for rnd in range(1, NUM_ROUNDS + 1):
        round_start = time.time()
        print(f"\n[ROUND {rnd}]")

        client_weights = []
        client_samples = []
        config = {"local_epochs": LOCAL_EPOCHS, "batch_size": BATCH_SIZE, "round": rnd}

        for cid, client in enumerate(clients):
            t0 = time.time()
            weights, n_samples, metrics = client.fit(global_weights, config)
            client_weights.append(weights)
            client_samples.append(n_samples)
            print(f"   Client {cid}: loss={metrics['loss']:.4f} "
                  f"acc={metrics['accuracy']:.4f} "
                  f"({n_samples} samples, {time.time()-t0:.1f}s)")

        # ── FedAvg aggregation (Eq. 2) ──
        global_weights = average_weights(client_weights, client_samples)

        round_time = time.time() - round_start
        round_times.append(round_time)
        peak_mem_mb = max(peak_mem_mb, psutil.Process().memory_info().rss / (1024 ** 2))
        print(f"   Round {rnd} aggregated in {round_time:.1f}s "
              f"(peak mem so far: {peak_mem_mb:.1f}MB)")

    # ── Build final model with the real aggregated weights ────────────────
    print("\n💾 Saving final model...")
    final_model = build_model(WINDOW_SIZE, n_features, num_classes)
    final_model.set_weights(global_weights)
    final_model.save("saved_models/fl_model.h5")
    print("✅ Model saved → saved_models/fl_model.h5 (real trained weights)")

    # ── Real evaluation on the held-out test set ───────────────────────────
    print("\n📊 Evaluating on test set...")
    y_pred = np.argmax(final_model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"   Test Accuracy: {acc:.4f}")
    print(f"   Macro F1:      {macro_f1:.4f}")

    # ── Overhead metrics ─────────────────────────────────────────────────
    model_size_mb = sum(w.nbytes for w in global_weights) / (1024 ** 2)
    comm_cost_per_round_mb = model_size_mb * NUM_CLIENTS * 2  # upload + download
    total_comm_cost_mb = comm_cost_per_round_mb * NUM_ROUNDS
    avg_round_time_s = sum(round_times) / len(round_times)

    print(f"\n📊 Overhead metrics:")
    print(f"   Model size: {model_size_mb:.2f} MB")
    print(f"   Comm. cost/round: {comm_cost_per_round_mb:.2f} MB")
    print(f"   Total comm. cost ({NUM_ROUNDS} rounds): {total_comm_cost_mb:.2f} MB")
    print(f"   Avg. round time: {avg_round_time_s:.2f} s")
    print(f"   Total training time: {sum(round_times):.2f} s")
    print(f"   Peak memory: {peak_mem_mb:.2f} MB")

    with open("saved_models/federated_run_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["accuracy", round(acc * 100, 2)])
        writer.writerow(["macro_f1", round(macro_f1 * 100, 2)])
        writer.writerow(["model_size_mb", round(model_size_mb, 2)])
        writer.writerow(["comm_cost_per_round_mb", round(comm_cost_per_round_mb, 2)])
        writer.writerow(["total_comm_cost_mb", round(total_comm_cost_mb, 2)])
        writer.writerow(["avg_round_time_s", round(avg_round_time_s, 2)])
        writer.writerow(["peak_mem_mb", round(peak_mem_mb, 2)])
    print("✅ Results saved → saved_models/federated_run_results.csv")

    print("\n📈 Round-by-round timing:")
    for rnd, t in enumerate(round_times, start=1):
        print(f"   Round {rnd:2d} → {t:.1f}s")

    return {"accuracy": acc, "macro_f1": macro_f1, "round_times": round_times}


if __name__ == "__main__":
    run()