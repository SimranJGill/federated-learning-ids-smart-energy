import sys
sys.stdout.reconfigure(line_buffering=True)
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pickle
import csv
import flwr as fl
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

from model.architecture import build_model
from federated.client import EnergyIoTClient
from federated.robust_strategy import (
    FedMedian, FedKrum, FedTrimmedMean, PoisonedClient
)
from blockchain_ledger import FLBlockchain

# ── Fixed config ─────────────────────────────────────────────────────
WINDOW_SIZE = 20
NUM_ROUNDS = 3
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

# Sample down for faster federated learning simulation
MAX_SIMULATION_SAMPLES = 2000
if len(X_all) > MAX_SIMULATION_SAMPLES:
    np.random.seed(42)  # for reproducibility
    idx = np.random.choice(len(X_all), MAX_SIMULATION_SAMPLES, replace=False)
    idx = np.sort(idx)
    X_all, y_all = X_all[idx], y_all[idx]

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

# ── One experiment run ───────────────────────────────────────────────
def run_experiment(strategy_name, num_poisoned, poison_scale, blockchain):
    """Trains one FL run and returns (accuracy, macro_f1) on the real test set."""

    def client_fn(cid: str):
        cid_int = int(cid)
        X, y = shards[cid_int]
        client = EnergyIoTClient(cid_int, X, y, num_classes)
        # Poison the FIRST `num_poisoned` clients (0..num_poisoned-1)
        if cid_int < num_poisoned:
            print(f"  Client {cid_int} ({CLIENT_NAMES[cid_int]}) is Byzantine "
                  f"(scale={poison_scale})")
            client = PoisonedClient(client, poison_scale=poison_scale)
        return client.to_client()

    strategy_kwargs = dict(
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS, min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=lambda rnd: {"local_epochs": 1, "batch_size": 64, "round": rnd},
    )

    class TrackedFedAvg(fl.server.strategy.FedAvg):
        def aggregate_fit(self, server_round, results, failures):
            params, metrics = super().aggregate_fit(server_round, results, failures)
            self.latest_parameters = params
            return params, metrics

    if strategy_name == "median":
        strategy = FedMedian(**strategy_kwargs)
    elif strategy_name == "krum":
        strategy = FedKrum(num_byzantine=max(num_poisoned, 1), **strategy_kwargs)
    elif strategy_name == "trimmed":
        strategy = FedTrimmedMean(beta=0.2, **strategy_kwargs)
    else:
        strategy = TrackedFedAvg(**strategy_kwargs)

    print(f"\n=== Strategy={strategy_name} | poisoned={num_poisoned} | "
          f"scale={poison_scale} ===")

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 0.0},
    )

    # --- Get the FINAL global weights out of the strategy, not a fresh model ---
    final_parameters = strategy.latest_parameters  # see note below
    final_ndarrays = fl.common.parameters_to_ndarrays(final_parameters)

    model = build_model(WINDOW_SIZE, n_features, num_classes)
    model.set_weights(final_ndarrays)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"  -> Test accuracy={acc:.4f}  Macro-F1={macro_f1:.4f}")

    # log a REAL block per round using history's distributed-fit metrics
    for rnd_idx, (rnd, loss) in enumerate(history.losses_distributed, start=1):
        blockchain.add_block(rnd, f"{strategy_name}_round{rnd}",
                              final_ndarrays[:1], acc)

    return acc, macro_f1

# ── Sweep across attack intensities ─────────────────────────────────
if __name__ == "__main__":
    import ray
    # Initialize Ray once for all experiments to prevent startup/shutdown timeouts on Windows
    ray.init(_node_ip_address="127.0.0.1", ignore_reinit_error=True)

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
        acc, f1 = run_experiment(strategy_name, num_poisoned, scale, blockchain)
        results.append({
            "strategy": strategy_name,
            "num_poisoned": num_poisoned,
            "poison_scale": scale,
            "accuracy": round(acc * 100, 2),
            "macro_f1": round(f1 * 100, 2),
        })

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

    # Shutdown Ray cluster session
    ray.shutdown()