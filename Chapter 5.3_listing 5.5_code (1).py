# Listing 5.5 — Communication overhead visualization from per-link CSV logs.

from __future__ import annotations

import csv
import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


@dataclass
class Row:
    t_s: float
    src: str
    dst: str
    overhead_bps: float


def load_rows(path: str) -> List[Row]:
    rows: List[Row] = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for line in r:
            rows.append(
                Row(
                    t_s=float(line["t_s"]),
                    src=line["src"],
                    dst=line["dst"],
                    overhead_bps=float(line["overhead_bytes_per_s"]),
                )
            )
    return rows


def plot_global_overhead(rows: List[Row], out_path: str) -> None:
    # Aggregate all links by timestamp (rounded to logger precision).
    by_t: Dict[float, float] = defaultdict(float)
    for row in rows:
        t = round(row.t_s, 6)
        by_t[t] += row.overhead_bps

    ts = sorted(by_t.keys())
    ys = [by_t[t] for t in ts]

    plt.figure()
    plt.plot(ts, ys)
    plt.xlabel("time (s)")
    plt.ylabel("overhead bytes/s")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)


def plot_top_links(rows: List[Row], out_path: str, k: int = 8) -> None:
    # Peak overhead rate per link (captures burst saturation risk).
    peaks: Dict[Tuple[str, str], float] = defaultdict(float)
    for row in rows:
        key = (row.src, row.dst)
        if row.overhead_bps > peaks[key]:
            peaks[key] = row.overhead_bps

    top = sorted(peaks.items(), key=lambda x: x[1], reverse=True)[:k]

    labels = [f"{src}->{dst}" for (src, dst), _ in top]
    vals = [v for _, v in top]

    plt.figure()
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), labels, rotation=45, ha="right")
    plt.ylabel("peak overhead bytes/s")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-global", default="overhead_global.png")
    ap.add_argument("--out-top", default="overhead_top_links.png")
    ap.add_argument("--top-k", type=int, default=8)
    args = ap.parse_args()

    rows = load_rows(args.csv)
    plot_global_overhead(rows, args.out_global)
    plot_top_links(rows, args.out_top, k=args.top_k)


if __name__ == "__main__":
    main()
