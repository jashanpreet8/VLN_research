import os
import json
import math
import argparse
import copy
import warnings
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


class R2RDataLoader:

    SPLIT_FILES = {
        "train":      "R2R_train.json",
        "val_seen":   "R2R_val_seen.json",
        "val_unseen": "R2R_val_unseen.json",
    }

    def __init__(self, data_dir: str, splits: Optional[List[str]] = None):
        self.data_dir = data_dir
        self.splits   = splits or list(self.SPLIT_FILES.keys())
        self.data: Dict[str, List[dict]] = {}
        self._loaded  = False

    def load(self) -> "R2RDataLoader":
        for split in self.splits:
            fname = os.path.join(self.data_dir, self.SPLIT_FILES[split])
            if not os.path.exists(fname):
                raise FileNotFoundError(f"Could not find: {fname}")
            with open(fname) as f:
                self.data[split] = json.load(f)
            print(f"  [R2RDataLoader] Loaded {split}: {len(self.data[split])} records")
        self._loaded = True
        return self

    @classmethod
    def from_synthetic(cls, n_records: int = 3000, seed: int = 42) -> "R2RDataLoader":
        rng    = np.random.default_rng(seed)
        loader = cls.__new__(cls)
        loader.data_dir = None
        loader.splits   = ["train", "val_seen", "val_unseen"]
        loader._loaded  = True

        scan_ids = [f"scan_{i:03d}" for i in range(90)]
        verbs    = ["walk", "go", "turn", "head", "exit", "enter", "pass", "move"]
        nouns    = ["door", "hallway", "staircase", "bedroom", "kitchen",
                    "bathroom", "living room", "couch", "table", "window"]
        preps    = ["to the left of", "past the", "toward the",
                    "next to the", "away from the"]

        def random_instruction(rng):
            parts = []
            for _ in range(rng.integers(2, 6)):
                parts.append(f"{rng.choice(verbs)} {rng.choice(preps)} {rng.choice(nouns)}")
            return ". ".join(parts) + "."

        records = []
        for pid in range(n_records):
            scan  = rng.choice(scan_ids)
            path  = [f"vp_{rng.integers(0, 9999):04d}" for _ in range(int(rng.integers(2, 10)))]
            dist  = float(rng.uniform(2.0, 30.0))
            records.append({
                "path_id":      pid,
                "scan":         scan,
                "path":         path,
                "instructions": [random_instruction(rng) for _ in range(3)],
                "heading":      float(rng.uniform(0, 2 * math.pi)),
                "distance":     dist,
            })

        rng.shuffle(records)
        n  = len(records)
        t  = int(0.70 * n)
        vs = int(0.85 * n)
        loader.data = {
            "train":      records[:t],
            "val_seen":   records[t:vs],
            "val_unseen": records[vs:],
        }
        print(f"  [R2RDataLoader] Synthetic dataset: train={t}, val_seen={vs-t}, val_unseen={n-vs}")
        return loader

    def all_records(self) -> List[dict]:
        out = []
        for split in self.splits:
            out.extend(self.data.get(split, []))
        return out

    def records_by_split(self, split: str) -> List[dict]:
        return self.data.get(split, [])

    def records_by_scan(self) -> Dict[str, List[dict]]:
        grouped = defaultdict(list)
        for rec in self.all_records():
            grouped[rec["scan"]].append(rec)
        return dict(grouped)


class R2REDA:

    def __init__(self, loader: R2RDataLoader, out_dir: str = "eda_outputs"):
        self.loader  = loader
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def _word_count(self, text: str) -> int:
        return len(text.split())

    def summary_statistics(self) -> dict:
        records       = self.loader.all_records()
        path_lengths  = [len(r["path"]) for r in records]
        distances     = [r["distance"]  for r in records]
        instr_lengths = [self._word_count(i) for r in records for i in r["instructions"]]
        scan_counts   = Counter(r["scan"] for r in records)

        stats = {
            "total_records"         : len(records),
            "unique_scans"          : len(scan_counts),
            "path_length_mean"      : float(np.mean(path_lengths)),
            "path_length_std"       : float(np.std(path_lengths)),
            "path_length_min"       : int(np.min(path_lengths)),
            "path_length_max"       : int(np.max(path_lengths)),
            "distance_mean_m"       : float(np.mean(distances)),
            "distance_std_m"        : float(np.std(distances)),
            "instr_word_mean"       : float(np.mean(instr_lengths)),
            "instr_word_std"        : float(np.std(instr_lengths)),
            "instr_word_min"        : int(np.min(instr_lengths)),
            "instr_word_max"        : int(np.max(instr_lengths)),
            "records_per_scan_mean" : float(np.mean(list(scan_counts.values()))),
        }

        print("\n" + "="*55)
        print("  R2R DATASET — SUMMARY STATISTICS")
        print("="*55)
        for k, v in stats.items():
            print(f"  {k:<35} {v:>10.2f}" if isinstance(v, float) else f"  {k:<35} {v:>10}")
        print("="*55 + "\n")
        return stats

    def _save(self, fig, name: str):
        path = os.path.join(self.out_dir, name)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  [EDA] Saved: {path}")

    def plot_path_length_distribution(self):
        records = self.loader.all_records()
        lengths = [len(r["path"]) for r in records]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(lengths, bins=range(1, max(lengths) + 2), color="#0d9488",
                edgecolor="white", linewidth=0.5, align="left")
        ax.set_xlabel("Number of Waypoints in Path", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("Distribution of Path Lengths (R2R)", fontsize=14, fontweight="bold")
        ax.axvline(np.mean(lengths), color="#ef4444", linestyle="--",
                   linewidth=1.5, label=f"Mean = {np.mean(lengths):.1f}")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        self._save(fig, "01_path_length_dist.png")

    def plot_instruction_length_distribution(self):
        records = self.loader.all_records()
        wc      = [len(i.split()) for r in records for i in r["instructions"]]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(wc, bins=30, color="#6366f1", edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Instruction Length (words)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("Distribution of Instruction Word Counts", fontsize=14, fontweight="bold")
        ax.axvline(np.mean(wc), color="#f59e0b", linestyle="--",
                   linewidth=1.5, label=f"Mean = {np.mean(wc):.1f} words")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        self._save(fig, "02_instr_length_dist.png")

    def plot_distance_distribution(self):
        records = self.loader.all_records()
        dists   = [r["distance"] for r in records]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(dists, bins=40, color="#f97316", edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Geodesic Distance (metres)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("Distribution of Navigation Distances", fontsize=14, fontweight="bold")
        ax.axvline(np.mean(dists), color="#3b82f6", linestyle="--",
                   linewidth=1.5, label=f"Mean = {np.mean(dists):.1f} m")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        self._save(fig, "03_distance_dist.png")

    def plot_scan_frequency(self):
        records = self.loader.all_records()
        counts  = Counter(r["scan"] for r in records)
        scans, freqs = zip(*sorted(counts.items(), key=lambda x: -x[1]))
        scans, freqs = scans[:20], freqs[:20]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(len(scans)), freqs, color=cm.viridis(np.linspace(0.3, 0.9, len(scans))))
        ax.set_xticks(range(len(scans)))
        ax.set_xticklabels(scans, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Building (Scan ID)", fontsize=12)
        ax.set_ylabel("Number of Paths", fontsize=12)
        ax.set_title("Top 20 Buildings by Number of Paths", fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        self._save(fig, "04_scan_frequency.png")

    def plot_path_length_vs_distance(self):
        records = self.loader.all_records()
        x = [len(r["path"]) for r in records]
        y = [r["distance"]  for r in records]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(x, y, alpha=0.3, s=12, color="#8b5cf6")
        m, b = np.polyfit(x, y, 1)
        xs   = np.linspace(min(x), max(x), 100)
        ax.plot(xs, m * xs + b, color="#ef4444", linewidth=2,
                label=f"Trend: y = {m:.2f}x + {b:.2f}")
        ax.set_xlabel("Path Length (# waypoints)", fontsize=12)
        ax.set_ylabel("Geodesic Distance (m)", fontsize=12)
        ax.set_title("Path Length vs. Navigation Distance", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        self._save(fig, "05_pathlen_vs_distance.png")

    def plot_split_overview(self):
        split_counts = {s: len(self.loader.records_by_split(s)) for s in self.loader.splits}
        labels, values = list(split_counts.keys()), list(split_counts.values())
        fig, ax = plt.subplots(figsize=(6, 5))
        bars = ax.bar(labels, values, color=["#0d9488", "#6366f1", "#f97316"][:len(labels)],
                      edgecolor="white", linewidth=0.8)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                    str(v), ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_ylabel("Number of Records", fontsize=12)
        ax.set_title("R2R Dataset Split Sizes", fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        self._save(fig, "06_split_overview.png")

    def run_all(self) -> dict:
        stats = self.summary_statistics()
        self.plot_path_length_distribution()
        self.plot_instruction_length_distribution()
        self.plot_distance_distribution()
        self.plot_scan_frequency()
        self.plot_path_length_vs_distance()
        self.plot_split_overview()
        print(f"\n  [EDA] All plots saved to '{self.out_dir}/'")
        return stats


class VLNFeatureExtractor:

    FEATURE_NAMES = [
        "instr_word_mean",
        "instr_word_std",
        "instr_char_mean",
        "unique_word_ratio",
        "heading_sin",
        "heading_cos",
        "distance",
    ]

    def transform(self, records: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for r in records:
            instrs   = r["instructions"]
            words    = [i.split() for i in instrs]
            wc       = [len(w) for w in words]
            cc       = [len(i) for i in instrs]
            all_w    = [w for ws in words for w in ws]
            unique_r = len(set(all_w)) / max(len(all_w), 1)
            X.append([
                float(np.mean(wc)),
                float(np.std(wc)),
                float(np.mean(cc)),
                unique_r,
                math.sin(r.get("heading", 0.0)),
                math.cos(r.get("heading", 0.0)),
                r.get("distance", 0.0),
            ])
            y.append(len(r["path"]))
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class VLNModel:

    def __init__(self, alpha: float = 1.0):
        self.alpha  = alpha
        self.scaler = StandardScaler()
        self.model  = Ridge(alpha=alpha)
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VLNModel":
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        self.fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X))

    def get_weights(self) -> dict:
        return {
            "coef"         : self.model.coef_.copy(),
            "intercept"    : float(self.model.intercept_),
            "scaler_mean"  : self.scaler.mean_.copy(),
            "scaler_scale" : self.scaler.scale_.copy(),
        }

    def set_weights(self, weights: dict) -> "VLNModel":
        self.model.coef_      = weights["coef"].copy()
        self.model.intercept_ = weights["intercept"]
        self.scaler.mean_     = weights["scaler_mean"].copy()
        self.scaler.scale_    = weights["scaler_scale"].copy()
        self.fitted           = True
        return self

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        preds = self.predict(X)
        return {
            "MAE" : float(mean_absolute_error(y, preds)),
            "RMSE": float(math.sqrt(mean_squared_error(y, preds))),
            "R2"  : float(r2_score(y, preds)),
        }


class FederatedClient:

    def __init__(self, client_id: str, records: List[dict],
                 extractor: VLNFeatureExtractor, alpha: float = 1.0):
        self.client_id = client_id
        self.records   = records
        self.extractor = extractor
        self.alpha     = alpha
        self.model     = VLNModel(alpha=alpha)
        self.n_samples = len(records)

    def local_train(self, global_weights: Optional[dict] = None) -> dict:
        X, y = self.extractor.transform(self.records)
        if global_weights is not None:
            self.model.set_weights(global_weights)
        else:
            self.model = VLNModel(alpha=self.alpha)
        self.model.fit(X, y)
        return self.model.get_weights()

    def evaluate_local(self) -> dict:
        X, y = self.extractor.transform(self.records)
        return self.model.evaluate(X, y)


class FederatedServer:

    def __init__(self, clients: List[FederatedClient],
                 rounds: int = 5, min_clients: Optional[int] = None):
        self.clients        = clients
        self.rounds         = rounds
        self.min_clients    = min_clients or len(clients)
        self.global_weights: Optional[dict] = None
        self.history: List[dict] = []

    def _fedavg(self, weight_list: List[dict], sizes: List[int]) -> dict:
        total = sum(sizes)
        avg   = {}
        for key in weight_list[0]:
            stacked  = np.stack([w[key] * (s / total) for w, s in zip(weight_list, sizes)])
            avg[key] = stacked.sum(axis=0)
        return avg

    def train(self) -> List[dict]:
        print("\n" + "="*55)
        print("  FEDERATED TRAINING")
        print("="*55)
        for rnd in range(1, self.rounds + 1):
            updated, sizes = [], []
            for client in self.clients[:self.min_clients]:
                updated.append(client.local_train(copy.deepcopy(self.global_weights)))
                sizes.append(client.n_samples)
            self.global_weights = self._fedavg(updated, sizes)
            maes = []
            for client in self.clients:
                client.model.set_weights(copy.deepcopy(self.global_weights))
                maes.append(client.evaluate_local()["MAE"])
            avg_mae = float(np.mean(maes))
            self.history.append({"round": rnd, "avg_mae": avg_mae})
            print(f"  Round {rnd:>2}/{self.rounds}  |  Global Avg MAE = {avg_mae:.4f}")
        print("="*55 + "\n")
        return self.history

    def evaluate_global(self, X: np.ndarray, y: np.ndarray) -> dict:
        if self.global_weights is None:
            raise RuntimeError("Model has not been trained yet.")
        model = VLNModel()
        model.set_weights(self.global_weights)
        return model.evaluate(X, y)

    def plot_training_history(self, out_path: str = "eda_outputs/07_federated_convergence.png"):
        if not self.history:
            print("  [Server] No training history to plot.")
            return
        rounds = [h["round"]   for h in self.history]
        maes   = [h["avg_mae"] for h in self.history]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(rounds, maes, marker="o", linewidth=2, color="#0d9488", markersize=7)
        ax.fill_between(rounds, maes, alpha=0.15, color="#0d9488")
        ax.set_xlabel("Communication Round", fontsize=12)
        ax.set_ylabel("Average MAE (path length)", fontsize=12)
        ax.set_title("Federated Learning Convergence (FedAvg)", fontsize=14, fontweight="bold")
        ax.grid(alpha=0.3)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Server] Convergence plot saved: {out_path}")


def run_pipeline(data_dir: Optional[str] = None):
    print("\n" + "#"*55)
    print("  VLN R2R — ANALYSIS & FEDERATED ML PIPELINE")
    print("#"*55 + "\n")

    print("[1/4] Loading data...")
    if data_dir and os.path.isdir(data_dir):
        loader = R2RDataLoader(data_dir).load()
    else:
        print("  No valid data_dir provided — using synthetic data.\n")
        loader = R2RDataLoader.from_synthetic(n_records=3000)

    print("\n[2/4] Running Exploratory Data Analysis...")
    eda = R2REDA(loader, out_dir="eda_outputs")
    eda.run_all()

    print("[3/4] Extracting features...")
    extractor    = VLNFeatureExtractor()
    X_all, y_all = extractor.transform(loader.all_records())
    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X_all, y_all, test_size=0.15, random_state=42
    )
    print(f"  Training pool: {len(X_train_all)} | Test: {len(X_test)}")

    print("\n[4/4] Building Federated Clients (one per building)...")
    scan_groups = loader.records_by_scan()
    clients = [
        FederatedClient(scan_id, recs, extractor, alpha=1.0)
        for scan_id, recs in scan_groups.items()
        if len(recs) >= 5
    ]
    print(f"  Active clients: {len(clients)} buildings")

    server = FederatedServer(clients, rounds=10)
    server.train()
    server.plot_training_history()

    metrics = server.evaluate_global(X_test, y_test)
    print("\n" + "="*55)
    print("  FINAL GLOBAL MODEL — TEST SET EVALUATION")
    print("="*55)
    for k, v in metrics.items():
        print(f"  {k:<10} {v:>8.4f}")
    print("="*55)

    if server.global_weights is not None:
        coefs = server.global_weights["coef"]
        print("\n  Feature importances (|coefficient| after standardisation):")
        for name, c in sorted(
            zip(VLNFeatureExtractor.FEATURE_NAMES, coefs),
            key=lambda x: abs(x[1]), reverse=True
        ):
            print(f"  {name:<30} {c:+.4f}  {'█' * int(abs(c) * 10)}")

    print("\n  Pipeline complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLN R2R Analysis & Federated ML")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to R2R annotation directory (optional; synthetic if omitted)")
    args = parser.parse_args()
    run_pipeline(args.data_dir)
