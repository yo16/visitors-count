"""定点カメラの動画から、一定間隔で静止画を切り出す。

ファイル名の開始時刻を基準に、interval 分おきのクリップを選び、
1 時点につき数秒ずらした複数枚を保存する。出力先は data/ 配下にすること
（人物が写るので公開リポジトリにコミットしない）。
"""
import argparse
import csv
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

CLIP_RE = re.compile(r"(\d{4})_(\d{2})_(\d{2})_(\d{6})_\d{2}\.MP4$", re.IGNORECASE)


def clip_start(path: Path) -> datetime | None:
    m = CLIP_RE.search(path.name)
    if not m:
        return None
    return datetime.strptime("".join(m.group(1, 2, 3, 4)), "%Y%m%d%H%M%S")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", type=Path, required=True, help="MP4 のあるディレクトリ")
    ap.add_argument("--out", type=Path, required=True, help="出力ディレクトリ")
    ap.add_argument("--start", default="08:46", help="切り出し開始 HH:MM")
    ap.add_argument("--end", default="13:29", help="切り出し終了 HH:MM（この時刻の分まで含む）")
    ap.add_argument("--interval", type=int, default=2, help="時点の間隔（分）")
    ap.add_argument("--offsets", type=float, nargs="+", default=[10.0, 13.0, 16.0],
                    help="クリップ先頭からの秒数。1 時点の枚数はこの個数")
    args = ap.parse_args()

    clips = sorted((clip_start(p), p) for p in args.clips.glob("*.MP4") if clip_start(p))
    if not clips:
        raise SystemExit(f"クリップが見つかりません: {args.clips}")

    day = clips[0][0].date()
    start = datetime.combine(day, datetime.strptime(args.start, "%H:%M").time())
    end = datetime.combine(day, datetime.strptime(args.end, "%H:%M").time()) + timedelta(minutes=1)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    next_t = start
    for t, path in clips:
        if t < next_t or t >= end:
            continue
        next_t = t.replace(second=0) + timedelta(minutes=args.interval)
        for i, offset in enumerate(args.offsets):
            shot = t + timedelta(seconds=offset)
            name = f"{t:%H%M%S}_{i}.jpg"
            dest = args.out / name
            if not dest.exists():
                subprocess.run(
                    ["ffmpeg", "-v", "error", "-y", "-ss", str(offset), "-i", str(path),
                     "-frames:v", "1", "-q:v", "2", str(dest)],
                    check=True,
                )
            rows.append({"file": name, "point": f"{t:%H%M%S}", "shot": i,
                         "time": f"{shot:%H:%M:%S}", "clip": path.name})

    with (args.out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "point", "shot", "time", "clip"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} 枚を {args.out} に保存しました")


if __name__ == "__main__":
    main()
