import os
import json
import math
import argparse
import warnings
from collections import defaultdict, Counter
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

    def load(self) -> "R2RDataLoader":
        for split in self.splits:
            fname = os.path.join(self.data_dir, self.SPLIT_FILES[split])
            if not os.path.exists(fname):
                raise FileNotFoundError(f"Could not find: {fname}")
            with open(fname) as f:
                self.data[split] = json.load(f)
            print(f"Loaded {split}: {len(self.data[split])} records")
        return self

    @classmethod
    def from_synthetic(cls, n_records: int = 3000, seed: int = 42) -> "R2RDataLoader":
        rng    = np.random.default_rng(seed)
        loader = cls.__new__(cls)
        loader.data_dir = None
        loader.splits   = ["train", "val_seen", "val_unseen"]

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
            scan = rng.choice(scan_ids)
            path = [f"vp_{rng.integers(0, 9999):04d}" for _ in range(int(rng.integers(2, 10)))]
            records.append({
                "path_id":      pid,
                "scan":         scan,
                "path":         path,
                "instructions": [random_instruction(rng) for _ in range(3)],
                "heading":      float(rng.uniform(0, 2 * math.pi)),
                "distance":     float(rng.uniform(2.0, 30.0)),
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
        print(f"Synthetic dataset: train={t}, val_seen={vs-t}, val_unseen={n-vs}")
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

    def summary_statistics(self):
        records       = self.loader.all_records()
        path_lengths  = [len(r["path"]) for r in records]
        distances     = [r["distance"]  for r in records]
        instr_lengths = [len(i.split()) for r in records for i in r["instructions"]]
        scan_counts   = Counter(r["scan"] for r in records)

        print(f"Total records:           {len(records)}")
        print(f"Unique buildings:        {len(scan_counts)}")
        print(f"Avg path length:         {np.mean(path_lengths):.2f} waypoints")
        print(f"Avg distance:            {np.mean(distances):.2f} m")
        print(f"Avg instruction length:  {np.mean(instr_lengths):.2f} words")

    def _save(self, fig, name: str):
        path = os.path.join(self.out_dir, name)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    def plot_path_length_distribution(self):
        records = self.loader.all_records()
        lengths = [len(r["path"]) for r in records]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(lengths, bins=range(1, max(lengths) + 2), color="#0d9488",
                edgecolor="white", align="left")
        ax.set_xlabel("Number of Waypoints")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Path Lengths")
        ax.axvline(np.mean(lengths), color="#ef4444", linestyle="--",
                   label=f"Mean = {np.mean(lengths):.1f}")
        ax.legend()
        self._save(fig, "01_path_length_dist.png")

    def plot_instruction_length_distribution(self):
        records = self.loader.all_records()
        wc      = [len(i.split()) for r in records for i in r["instructions"]]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(wc, bins=30, color="#6366f1", edgecolor="white")
        ax.set_xlabel("Word Count")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Instruction Lengths")
        ax.axvline(np.mean(wc), color="#f59e0b", linestyle="--",
                   label=f"Mean = {np.mean(wc):.1f}")
        ax.legend()
        self._save(fig, "02_instr_length_dist.png")

    def plot_distance_distribution(self):
        records = self.loader.all_records()
        dists   = [r["distance"] for r in records]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(dists, bins=40, color="#f97316", edgecolor="white")
        ax.set_xlabel("Distance (metres)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Navigation Distances")
        ax.axvline(np.mean(dists), color="#3b82f6", linestyle="--",
                   label=f"Mean = {np.mean(dists):.1f} m")
        ax.legend()
        self._save(fig, "03_distance_dist.png")

    def plot_scan_frequency(self):
        records = self.loader.all_records()
        counts  = Counter(r["scan"] for r in records)
        scans, freqs = zip(*sorted(counts.items(), key=lambda x: -x[1]))
        scans, freqs = scans[:20], freqs[:20]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(len(scans)), freqs,
               color=cm.viridis(np.linspace(0.3, 0.9, len(scans))))
        ax.set_xticks(range(len(scans)))
        ax.set_xticklabels(scans, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Building (Scan ID)")
        ax.set_ylabel("Number of Paths")
        ax.set_title("Top 20 Buildings by Number of Paths")
        self._save(fig, "04_scan_frequency.png")

    def plot_path_length_vs_distance(self):
        records = self.loader.all_records()
        x = [len(r["path"]) for r in records]
        y = [r["distance"]  for r in records]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(x, y, alpha=0.3, s=12, color="#8b5cf6")
        ax.set_xlabel("Path Length (waypoints)")
        ax.set_ylabel("Distance (m)")
        ax.set_title("Path Length vs. Navigation Distance")
        self._save(fig, "05_pathlen_vs_distance.png")

    def plot_split_overview(self):
        split_counts = {s: len(self.loader.records_by_split(s)) for s in self.loader.splits}
        labels, values = list(split_counts.keys()), list(split_counts.values())
        fig, ax = plt.subplots(figsize=(6, 5))
        bars = ax.bar(labels, values,
                      color=["#0d9488", "#6366f1", "#f97316"][:len(labels)])
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                    str(v), ha="center", fontweight="bold")
        ax.set_ylabel("Number of Records")
        ax.set_title("Dataset Split Sizes")
        self._save(fig, "06_split_overview.png")

    def run_all(self):
        self.summary_statistics()
        self.plot_path_length_distribution()
        self.plot_instruction_length_distribution()
        self.plot_distance_distribution()
        self.plot_scan_frequency()
        self.plot_path_length_vs_distance()
        self.plot_split_overview()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()

    if args.data_dir and os.path.isdir(args.data_dir):
        loader = R2RDataLoader(args.data_dir).load()
    else:
        loader = R2RDataLoader.from_synthetic(n_records=3000)

    eda = R2REDA(loader)
    eda.run_all()
