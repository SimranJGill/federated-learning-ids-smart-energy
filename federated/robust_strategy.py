import numpy as np
import flwr as fl
from flwr.common import (
    Parameters, Scalar, FitRes, EvaluateRes,
    parameters_to_ndarrays, ndarrays_to_parameters,
)
from typing import List, Tuple, Optional, Dict, Union
from functools import reduce


# ── Helper: flatten / unflatten weights ──────────────────────────────
def flatten_weights(weights: List[np.ndarray]) -> np.ndarray:
    return np.concatenate([w.flatten() for w in weights])

def unflatten_weights(flat: np.ndarray,
                      shapes: List[tuple]) -> List[np.ndarray]:
    weights, idx = [], 0
    for shape in shapes:
        size = int(np.prod(shape))
        weights.append(flat[idx:idx+size].reshape(shape))
        idx += size
    return weights


# ══════════════════════════════════════════════════════════════════════
# STRATEGY 1 — COORDINATE-WISE MEDIAN
# Most robust against Byzantine clients — even if 49% are malicious
# ══════════════════════════════════════════════════════════════════════
class FedMedian(fl.server.strategy.FedAvg):
    """
    Byzantine-robust aggregation using coordinate-wise median.
    Replaces simple mean with median — resistant to outlier weights
    from malicious or compromised IoT clients.
    """

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}

        weights_list = [
            parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]

        print(f"\n[FedMedian] Round {server_round} — "
              f"aggregating {len(weights_list)} clients")

        # Coordinate-wise median across all clients
        median_weights = [
            np.median(
                np.stack([w[i] for w in weights_list], axis=0),
                axis=0
            )
            for i in range(len(weights_list[0]))
        ]

        parameters_aggregated = ndarrays_to_parameters(median_weights)
        metrics_aggregated    = {}
        self.latest_parameters = parameters_aggregated
        return parameters_aggregated, metrics_aggregated


# ══════════════════════════════════════════════════════════════════════
# STRATEGY 2 — KRUM
# Selects the client whose weights are closest to all others
# Best when there are known Byzantine clients
# ══════════════════════════════════════════════════════════════════════
class FedKrum(fl.server.strategy.FedAvg):
    """
    Krum Byzantine-robust aggregation.
    Selects the single client update that is closest
    (in Euclidean distance) to all other updates.
    Resistant to up to (n/2 - 1) Byzantine clients.
    """

    def __init__(self, num_byzantine: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.num_byzantine = num_byzantine

    def _krum_scores(self,
                     flat_weights: List[np.ndarray],
                     num_byzantine: int) -> np.ndarray:
        n  = len(flat_weights)
        m  = n - num_byzantine - 2  # number of neighbors to consider
        m  = max(1, m)

        # Compute pairwise squared distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = np.sum((flat_weights[i] - flat_weights[j]) ** 2)
                distances[i][j] = d
                distances[j][i] = d

        # For each client, sum distances to its m closest neighbors
        scores = np.zeros(n)
        for i in range(n):
            sorted_d = np.sort(distances[i])
            scores[i] = np.sum(sorted_d[1:m+1])  # exclude self (0)
        return scores

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}

        weights_list = [
            parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]
        shapes       = [w.shape for w in weights_list[0]]
        flat_list    = [flatten_weights(w) for w in weights_list]

        scores = self._krum_scores(flat_list, self.num_byzantine)
        best   = int(np.argmin(scores))

        print(f"\n[FedKrum] Round {server_round} — "
              f"selected client {best} (score={scores[best]:.4f})")
        print(f"  All scores: {[f'{s:.2f}' for s in scores]}")

        # Use the best client's weights as global update
        selected_weights = weights_list[best]
        parameters_aggregated = ndarrays_to_parameters(selected_weights)
        self.latest_parameters = parameters_aggregated
        return parameters_aggregated, {}


# ══════════════════════════════════════════════════════════════════════
# STRATEGY 3 — TRIMMED MEAN
# Remove top and bottom X% of weights before averaging
# Good balance between robustness and accuracy
# ══════════════════════════════════════════════════════════════════════
class FedTrimmedMean(fl.server.strategy.FedAvg):
    """
    Byzantine-robust aggregation using trimmed mean.
    Removes the highest and lowest beta fraction of values
    per coordinate before averaging.
    """

    def __init__(self, beta: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta  # fraction to trim from each end

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}

        weights_list = [
            parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]

        n        = len(weights_list)
        trim_k   = max(1, int(n * self.beta))

        print(f"\n[TrimmedMean] Round {server_round} — "
              f"{n} clients, trimming {trim_k} from each end")

        trimmed_weights = []
        for i in range(len(weights_list[0])):
            stacked = np.stack([w[i] for w in weights_list], axis=0)
            # Sort and trim along client axis (axis=0)
            sorted_w = np.sort(stacked, axis=0)
            trimmed  = sorted_w[trim_k : n - trim_k]
            trimmed_weights.append(np.mean(trimmed, axis=0))

        parameters_aggregated = ndarrays_to_parameters(trimmed_weights)
        self.latest_parameters = parameters_aggregated
        return parameters_aggregated, {}


# ══════════════════════════════════════════════════════════════════════
# POISONING ATTACK SIMULATOR
# Makes one client send corrupted random weights
# Use this to test robustness of your strategies
# ══════════════════════════════════════════════════════════════════════
class PoisonedClient(fl.client.NumPyClient):
    """
    Wraps a normal client and injects poisoned weights.
    Use to simulate a Byzantine/compromised IoT device.
    """

    def __init__(self, normal_client, poison_scale: float = 10.0):
        self.client       = normal_client
        self.poison_scale = poison_scale

    def get_parameters(self, config):
        return self.client.get_parameters(config)

    def fit(self, parameters, config):
        weights, num_samples, metrics = self.client.fit(
            parameters, config)
        # Poison: replace weights with scaled random noise
        poisoned = [
            np.random.randn(*w.shape) * self.poison_scale
            for w in weights
        ]
        print(f"  ☠️  Poisoned client sending corrupted weights!")
        return poisoned, num_samples, {"poisoned": True}

    def evaluate(self, parameters, config):
        return self.client.evaluate(parameters, config)


# ══════════════════════════════════════════════════════════════════════
# DEMO — compare FedAvg vs FedMedian vs FedKrum
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Byzantine-Robust Aggregation Demo")
    print("="*50)

    # Simulate 5 normal clients + 1 Byzantine
    normal_weights = [np.array([1.0, 2.0, 3.0])] * 4
    byzantine_w    = [np.array([100.0, -100.0, 200.0])]
    all_weights    = normal_weights + byzantine_w

    print("Client weights (simplified 1D demo):")
    for i, w in enumerate(all_weights):
        label = "☠️ Byzantine" if i == 4 else "✅ Normal"
        print(f"  Client {i}: {w[0]}  {label}")

    # FedAvg
    fedavg_result = np.mean([w[0] for w in all_weights])
    print(f"\n❌ FedAvg result     : {fedavg_result:.2f}  (skewed!)")

    # Median
    median_result = np.median([w[0] for w in all_weights])
    print(f"✅ FedMedian result  : {median_result:.2f}  (robust!)")

    # Trimmed Mean (trim 1 from each end)
    sorted_vals   = sorted([w[0] for w in all_weights])
    trimmed       = sorted_vals[1:-1]
    trimmed_result= np.mean(trimmed)
    print(f"✅ TrimmedMean result: {trimmed_result:.2f}  (robust!)")

    print("\nConclusion: FedMedian and TrimmedMean ignore the Byzantine client.")