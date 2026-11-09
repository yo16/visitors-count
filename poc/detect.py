"""静止画から人物を検出し、検出結果を CSV に保存する。

全体 1 回の推論（full）と、画像を重ねて分割した推論を統合するもの（tiled）を比較できる。
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.ops import nms
from ultralytics import YOLO

PERSON = 0


def tiles(width: int, height: int, size: int, overlap: int, top: int):
    step = size - overlap
    xs = list(range(0, max(width - size, 0) + 1, step))
    if xs[-1] != width - size:
        xs.append(width - size)
    ys = list(range(top, max(height - size, top) + 1, step))
    if ys[-1] != height - size:
        ys.append(height - size)
    return [(x, y) for y in ys for x in xs]


def predict(model, image, imgsz: int, conf: float):
    result = model.predict(image, imgsz=imgsz, conf=conf, classes=[PERSON], verbose=False, device="cpu")[0]
    if result.boxes is None or len(result.boxes) == 0:
        return np.zeros((0, 5))
    return np.hstack([result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()[:, None]])


def detect_image(model, image, mode: str, args) -> np.ndarray:
    boxes = [predict(model, image, args.imgsz, args.conf)]
    if mode == "tiled":
        h, w = image.shape[:2]
        for x, y in tiles(w, h, args.tile, args.overlap, args.top):
            found = predict(model, image[y:y + args.tile, x:x + args.tile], args.tile_imgsz, args.conf)
            found[:, [0, 2]] += x
            found[:, [1, 3]] += y
            boxes.append(found)
    merged = np.vstack(boxes)
    if len(merged) == 0:
        return merged
    keep = nms(torch.tensor(merged[:, :4], dtype=torch.float32),
               torch.tensor(merged[:, 4], dtype=torch.float32), args.iou).numpy()
    return merged[keep]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="検出結果 CSV のパス")
    ap.add_argument("--model", default="data/models/yolo11m.pt")
    ap.add_argument("--mode", choices=["full", "tiled"], default="tiled")
    ap.add_argument("--only", nargs="*", default=None, help="時点（HHMMSS）の前方一致で絞る")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--tile-imgsz", type=int, default=960)
    ap.add_argument("--overlap", type=int, default=120)
    ap.add_argument("--top", type=int, default=400, help="タイルを敷く上端の y（空を避ける）")
    ap.add_argument("--overlay-dir", type=Path, default=None, help="確認用の描画画像の出力先")
    args = ap.parse_args()

    manifest = pd.read_csv(args.frames / "manifest.csv", dtype={"point": str})
    if args.only:
        manifest = manifest[manifest["point"].str.startswith(tuple(args.only))]

    model = YOLO(args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.overlay_dir:
        args.overlay_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "point", "shot", "x0", "y0", "x1", "y1", "conf"])
        for done, row in enumerate(manifest.itertuples(), start=1):
            image = cv2.imread(str(args.frames / row.file))
            found = detect_image(model, image, args.mode, args)
            for x0, y0, x1, y1, conf in found:
                writer.writerow([row.file, row.point, row.shot,
                                 round(float(x0), 1), round(float(y0), 1),
                                 round(float(x1), 1), round(float(y1), 1), round(float(conf), 3)])
            f.flush()
            if args.overlay_dir and row.shot == 0:
                canvas = image.copy()
                for x0, y0, x1, y1, _ in found:
                    cv2.rectangle(canvas, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
                cv2.imwrite(str(args.overlay_dir / f"{args.mode}_{row.file}"), canvas)
            elapsed = time.time() - started
            print(f"{done}/{len(manifest)} {row.file} {len(found)} 人 ({elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
