import sys
sys.stdout.reconfigure(line_buffering=True)
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import numpy as np
import pickle
import csv
import psutil
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

from model.architecture import build_model
from federated.client import EnergyIoTClient
from blockchain_ledger import FLBlockchain

# ── Fixed config ─────────────────────────────────────────────────────
WINDOW_SIZE = 20
NUM_ROUNDS = 10
NUM_CLIENTS = 5
CLIENT_NAMES = ["SmartHome", "EVCharging", "GridSensor",
                "SolarWind", "IndustrialEnergy"]

# ── Load data once ───────────────────────────────────────────────────
print("Loading sequence data...")
X_cic = np.load("data/processed/X_seq_cic.npy")
y_cic = np.load("data/processed/y_seq_cic.npy")
X_edge = np.load("data/processed/X_seq_edge.npy")
y_edge = np.load("data/processed/y_seq_edge.npy")
with open("data/processed/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

X_all = np.concatenate([X_cic, X_edge], axis=0)
y_all = np.concatenate([y_cic, y_edge], axis=0)

keep_mask = np.zeros(len(y_all), dtype=bool)
for i in range(len(le.classes_)):
    if np.sum(y_all == i) >= 10:
        keep_mask |= (y_all == i)
X_all, y_all = X_all[keep_mask], y_all[keep_mask]

unique = np.unique(y_all)
remap = {old: new for new, old in enumerate(unique)}
y_all = np.array([remap[y] for y in y_all])
num_classes = len(unique)
n_features = X_all.shape[2]

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.15, random_state=42)


def partition_data(X, y, n=5):
    idx = np.array_split(np.arange(len(X)), n)
    return [(X[i], y[i]) for i in idx]


shards = partition_data(X_train, y_train, NUM_CLIENTS)


# ── Aggregation Strategies ───────────────────────────────────────────
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


def median_aggregation(weights_list):
    """Coordinate-wise median aggregation."""
    n_layers = len(weights_list[0])
    averaged = []
    for layer_idx in range(n_layers):
        stacked = np.stack([w[layer_idx] for w in weights_list], axis=0)
        averaged.append(np.median(stacked, axis=0))
    return averaged


def flatten_weights(weights):
    return np.concatenate([w.flatten() for w in weights])


def krum_aggregation(weights_list, num_byzantine=1):
    """Krum Byzantine-robust selection."""
    n_clients = len(weights_list)
    m = n_clients - num_byzantine - 2
    m = max(1, m)

    flat_weights = [flatten_weights(w) for w in weights_list]
    distances = np.zeros((n_clients, n_clients))
    for i in range(n_clients):
        for j in range(i+1, n_clients):
            d = np.sum((flat_weights[i] - flat_weights[j]) ** 2)
            distances[i][j] = d
            distances[j][i] = d

    scores = np.zeros(n_clients)
    for i in range(n_clients):
        sorted_d = np.sort(distances[i])
        scores[i] = np.sum(sorted_d[1:m+1])

    best = int(np.argmin(scores))
    print(f"   [FedKrum] selected client {best} with score {scores[best]:.4f}")
    return weights_list[best]


def trimmed_mean_aggregation(weights_list, beta=0.2):
    """Coordinate-wise trimmed mean aggregation."""
    n_clients = len(weights_list)
    trim_k = max(1, int(n_clients * beta))
    n_layers = len(weights_list[0])
    averaged = []
    for layer_idx in range(n_layers):
        stacked = np.stack([w[layer_idx] for w in weights_list], axis=0)
        sorted_w = np.sort(stacked, axis=0)
        trimmed = sorted_w[trim_k : n_clients - trim_k]
        averaged.append(np.mean(trimmed, axis=0))
    return averaged


# ── One experiment run ───────────────────────────────────────────────
def run_experiment(strategy_name, num_poisoned, poison_scale, blockchain):
    """Trains one FL run and returns real metrics from the actual trained model."""
    print(f"\n=== Strategy={strategy_name} | poisoned={num_poisoned} | scale={poison_scale} ===")

    # Re-initialize client models for each experiment to start from fresh weights
    clients = []
    for cid, (X, y) in enumerate(shards):
        clients.append(EnergyIoTClient(cid, X, y, num_classes))

    # Global weights initialized from client 0
    global_weights = clients[0].get_parameters(config={})

    round_times = []
    peak_mem_mb = 0.0

    for rnd in range(1, NUM_ROUNDS + 1):
        round_start = time.time()

        client_weights = []
        client_samples = []
        config = {"local_epochs": 3, "batch_size": 64, "round": rnd}

        for cid, client in enumerate(clients):
            # Normal local training
            weights, n_samples, metrics = client.fit(global_weights, config)

            # In-process Byzantine poisoning simulation
            if cid < num_poisoned:
                weights = [
                    np.random.randn(*w.shape) * poison_scale
                    for w in weights
                ]
                print(f"   Round {rnd} - Client {cid} ({CLIENT_NAMES[cid]}): Byzantine (noise scale={poison_scale})")
            else:
                print(f"   Round {rnd} - Client {cid} ({CLIENT_NAMES[cid]}): loss={metrics['loss']:.4f} acc={metrics['accuracy']:.4f}")

            client_weights.append(weights)
            client_samples.append(n_samples)

        # ── Global Aggregation ──
        if strategy_name == "median":
            global_weights = median_aggregation(client_weights)
        elif strategy_name == "krum":
            global_weights = krum_aggregation(client_weights, num_byzantine=max(num_poisoned, 1))
        elif strategy_name == "trimmed":
            global_weights = trimmed_mean_aggregation(client_weights, beta=0.2)
        else:
            global_weights = average_weights(client_weights, client_samples)

        round_time = time.time() - round_start
        round_times.append(round_time)
        peak_mem_mb = max(peak_mem_mb, psutil.Process().memory_info().rss / (1024 ** 2))

    # Evaluate final aggregated weights on held-out test set
    model = build_model(WINDOW_SIZE, n_features, num_classes)
    model.set_weights(global_weights)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    # Overhead metrics
    model_size_mb = sum(w.nbytes for w in global_weights) / (1024 ** 2)
    comm_cost_per_round_mb = model_size_mb * NUM_CLIENTS * 2
    total_comm_cost_mb = comm_cost_per_round_mb * NUM_ROUNDS
    avg_round_time_s = sum(round_times) / len(round_times)
    total_train_time_s = sum(round_times)

    print(f"  -> Test accuracy={acc:.4f}  Macro-F1={macro_f1:.4f}")
    print(f"  -> Model size={model_size_mb:.2f}MB  "
          f"Comm/round={comm_cost_per_round_mb:.2f}MB  "
          f"Avg round time={avg_round_time_s:.2f}s  "
          f"Peak mem={peak_mem_mb:.2f}MB")

    # Real blockchain entry
    for rnd in range(1, NUM_ROUNDS + 1):
        blockchain.add_block(rnd, f"{strategy_name}_p{num_poisoned}_s{poison_scale}",
                              global_weights, acc)

    return {
        "strategy": strategy_name,
        "num_poisoned": num_poisoned,
        "poison_scale": poison_scale,
        "accuracy": round(acc * 100, 2),
        "macro_f1": round(macro_f1 * 100, 2),
        "model_size_mb": round(model_size_mb, 2),
        "comm_cost_per_round_mb": round(comm_cost_per_round_mb, 2),
        "total_comm_cost_mb": round(total_comm_cost_mb, 2),
        "avg_round_time_s": round(avg_round_time_s, 2),
        "total_train_time_s": round(total_train_time_s, 2),
        "peak_mem_mb": round(peak_mem_mb, 2),
    }


# ── Sweep across attack intensities ─────────────────────────────────
if __name__ == "__main__":
    blockchain = FLBlockchain()
    results = []

    configs = [
        # (strategy, num_poisoned, poison_scale)
        ("fedavg", 0, 0.0),
        ("median", 0, 0.0),
        ("fedavg", 1, 5.0),
        ("median", 1, 5.0),
        ("fedavg", 2, 5.0),
        ("median", 2, 5.0),
        ("fedavg", 3, 5.0),
        ("median", 3, 5.0),
        ("median", 1, 2.0),
    ]

    for strategy_name, num_poisoned, scale in configs:
        results.append(run_experiment(strategy_name, num_poisoned, scale, blockchain))

    os.makedirs("saved_models", exist_ok=True)
    with open("saved_models/robustness_sweep_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    blockchain.print_chain()
    blockchain.export_to_json("saved_models/blockchain_ledger.json")

    print("\n===== SUMMARY =====")
    for r in results:
        print(r)