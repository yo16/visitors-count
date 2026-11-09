"""密度推定モデル（lwcc）で静止画の人数を推定し、CSV と確認画像を保存する。

lwcc は Windows で重みの保存先を誤るため、weights_check を差し替えて --weights-dir に置く。
"""
import argparse
import csv
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from lwcc import LWCC
from lwcc.models import Bay, CSRNet, DMCount, SFANet

URL = "https://github.com/tersekmatija/lwcc_weights/releases/download/v0.1/{}"


def use_weights_dir(weights_dir: Path) -> None:
    def weights_check(model_name: str, model_weights: str) -> str:
        name = f"{model_name}_{model_weights}.pth"
        path = weights_dir / name
        if not path.is_file():
            weights_dir.mkdir(parents=True, exist_ok=True)
            print(f"download {name}", flush=True)
            urllib.request.urlretrieve(URL.format(name), path)
        return str(path)

    for module in (Bay, CSRNet, DMCount, SFANet):
        module.weights_check = weights_check


def overlay(image: np.ndarray, density: np.ndarray) -> np.ndarray:
    resized = cv2.resize(density, (image.shape[1], image.shape[0]))
    scale = max(float(np.percentile(resized, 99.5)), 1e-9)
    heat = cv2.applyColorMap(np.uint8(np.clip(resized / scale, 0, 1) * 255), cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 0.55, heat, 0.45, 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="結果 CSV のパス")
    ap.add_argument("--models", nargs="+", default=["DM-Count:SHA"], help="モデル:重み の並び")
    ap.add_argument("--weights-dir", type=Path, default=Path("data/models/lwcc"))
    ap.add_argument("--only", nargs="*", default=None, help="時点（HHMMSS）の前方一致で絞る")
    ap.add_argument("--shots", type=int, nargs="*", default=[0], help="使う連番")
    ap.add_argument("--overlay-dir", type=Path, default=None, help="確認用のヒートマップの出力先")
    args = ap.parse_args()

    use_weights_dir(args.weights_dir)
    manifest = pd.read_csv(args.frames / "manifest.csv", dtype={"point": str})
    manifest = manifest[manifest["shot"].isin(args.shots)]
    if args.only:
        manifest = manifest[manifest["point"].str.startswith(tuple(args.only))]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.overlay_dir:
        args.overlay_dir.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "point", "shot", "model", "count", "seconds"])
        for spec in args.models:
            name, weights = spec.split(":")
            model = LWCC.load_model(name, weights)
            for row in manifest.itertuples():
                path = args.frames / row.file
                started = time.time()
                count, density = LWCC.get_count(str(path), model=model, return_density=True, resize_img=False)
                seconds = time.time() - started
                writer.writerow([row.file, row.point, row.shot, spec, round(float(count), 1), round(seconds, 1)])
                f.flush()
                print(f"{spec} {row.file} {float(count):.0f} 人 ({seconds:.1f}s)", flush=True)
                if args.overlay_dir:
                    label = spec.replace(":", "_")
                    cv2.imwrite(str(args.overlay_dir / f"{label}_{row.file}"), overlay(cv2.imread(str(path)), density))


if __name__ == "__main__":
    main()
