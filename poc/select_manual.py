"""密度推定の結果を時点ごとに集計し、手動で点を打つ時点を混雑度で層別して選ぶ。"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--counts", type=Path, required=True, help="density.py の結果 CSV")
    ap.add_argument("--out", type=Path, required=True, help="出力先フォルダ")
    ap.add_argument("--exclude", nargs="*", default=["132809"], help="除外する時点（HHMMSS）")
    ap.add_argument("--strata", type=int, default=6)
    ap.add_argument("--per-stratum", type=int, default=5)
    args = ap.parse_args()

    counts = pd.read_csv(args.counts, dtype={"point": str})
    counts = counts[~counts["point"].isin(args.exclude)]
    manifest = pd.read_csv(args.frames / "manifest.csv", dtype={"point": str})
    times = manifest[manifest["shot"] == 0].set_index("point")["time"]

    points = counts.groupby("point")["count"].agg(["median", "min", "max"]).reset_index()
    points["time"] = points["point"].map(times)
    points = points.sort_values("time")
    points["stratum"] = pd.qcut(points["median"], args.strata, labels=False)

    picks = []
    for _, group in points.groupby("stratum"):
        indices = np.linspace(0, len(group) - 1, args.per_stratum).round().astype(int)
        picks.append(group.iloc[np.unique(indices)])
    selected = pd.concat(picks).sort_values("time")

    images = args.out / "images"
    images.mkdir(parents=True, exist_ok=True)
    for point in selected["point"]:
        shutil.copy2(args.frames / f"{point}_0.jpg", images / f"{point}_0.jpg")
    points.to_csv(args.out / "curve_by_point.csv", index=False)
    selected.to_csv(args.out / "selection.csv", index=False)

    points["slot"] = points["time"].str[:2] + ":" + points["time"].str[3:5].astype(int).floordiv(30).mul(30).astype(str).str.zfill(2)
    print(points.groupby("slot")["median"].agg(["count", "median", "min", "max"]).round(0).to_string())
    print(f"\n選んだ時点: {len(selected)}")
    print(selected[["time", "median", "stratum"]].to_string(index=False))
    spread = (points["max"] - points["min"]) / points["median"].clip(lower=1)
    print(f"\n同じ時点の 3 枚のばらつき（幅/中央値）の中央値: {spread[points['median'] >= 50].median():.2f}")


if __name__ == "__main__":
    main()
