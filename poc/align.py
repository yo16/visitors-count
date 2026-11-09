"""各時点のフレームを基準フレームへ位置合わせする。

人が写らない静止物（画面上部のビルや建物）の特徴点だけを使い、
回転、拡大縮小、平行移動（相似変換）を推定して CSV に保存する。
確認用に、基準との重ね合わせ画像も出力する。
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# 静止物を探す領域（画像に対する比率で x0, y0, x1, y1）
STATIC_REGIONS = [(0.0, 0.10, 0.20, 0.60), (0.55, 0.25, 1.0, 0.62)]


def static_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    for x0, y0, x1, y1 in STATIC_REGIONS:
        mask[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 255
    return mask


def detect(orb, gray: np.ndarray):
    return orb.detectAndCompute(gray, static_mask(gray.shape))


def estimate(kp, des, ref_kp, ref_des):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    good = []
    for pair in matcher.knnMatch(des, ref_des, k=2):
        if len(pair) == 2 and pair[0].distance < 0.8 * pair[1].distance:
            good.append(pair[0])
    if len(good) < 8:
        return None, 0, len(good)
    src = np.float32([kp[m.queryIdx].pt for m in good])
    dst = np.float32([ref_kp[m.trainIdx].pt for m in good])
    matrix, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=5000, confidence=0.999
    )
    if matrix is None:
        return None, 0, len(good)
    return matrix, int(inliers.sum()), len(good)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="alignment.csv の出力先ディレクトリ")
    ap.add_argument("--ref-time", default="110000", help="基準にする時点（HHMMSS）。以降で最初の時点を使う")
    ap.add_argument("--qa-every", type=int, default=20, help="確認画像を何時点ごとに出すか")
    args = ap.parse_args()

    manifest = pd.read_csv(args.frames / "manifest.csv", dtype={"point": str})
    points = sorted(manifest["point"].unique())
    ref_point = next(p for p in points if p >= args.ref_time)
    ref_img = cv2.imread(str(args.frames / f"{ref_point}_0.jpg"))
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=8000, fastThreshold=7)
    ref_kp, ref_des = detect(orb, ref_gray)

    args.out.mkdir(parents=True, exist_ok=True)
    qa_dir = args.out / "qa"
    qa_dir.mkdir(exist_ok=True)
    h, w = ref_gray.shape

    rows = []
    for idx, point in enumerate(points):
        img = cv2.imread(str(args.frames / f"{point}_0.jpg"))
        kp, des = detect(orb, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        matrix, inliers, matches = estimate(kp, des, ref_kp, ref_des)
        row = {"point": point, "inliers": inliers, "matches": matches, "ok": matrix is not None}
        if matrix is not None:
            a, b = matrix[0, 0], matrix[1, 0]
            row.update({
                "m00": matrix[0, 0], "m01": matrix[0, 1], "m02": matrix[0, 2],
                "m10": matrix[1, 0], "m11": matrix[1, 1], "m12": matrix[1, 2],
                "scale": float(np.hypot(a, b)),
                "rot_deg": float(np.degrees(np.arctan2(b, a))),
                "dx": float(matrix[0, 2]), "dy": float(matrix[1, 2]),
            })
            if idx % args.qa_every == 0 or idx == len(points) - 1:
                warped = cv2.warpAffine(img, matrix, (w, h))
                blend = cv2.addWeighted(ref_img, 0.5, warped, 0.5, 0)
                cv2.imwrite(str(qa_dir / f"blend_{point}.jpg"), cv2.resize(blend, (960, 540)))
        rows.append(row)

    with (args.out / "alignment.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["point", "ok", "inliers", "matches", "scale", "rot_deg", "dx", "dy",
                  "m00", "m01", "m02", "m10", "m11", "m12"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    df = pd.DataFrame(rows)
    print(f"基準: {ref_point} / 成功 {int(df['ok'].sum())}/{len(df)}")
    print(df[["point", "inliers", "matches", "scale", "rot_deg", "dx", "dy"]].iloc[::12].to_string())
    print(df[["point", "inliers", "matches", "scale", "rot_deg", "dx", "dy"]].tail(4).to_string())


if __name__ == "__main__":
    main()
