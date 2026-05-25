#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


VIDEO_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left",
    "observation.images.cam_right",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a RoboTwin dataset copy with resized H264 videos."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("/data/datasets/dreamzero/robotwin_unified_dreamzero"),
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("/data/datasets/dreamzero/robotwin_unified_dreamzero_180x320"),
    )
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--ffmpeg", default="/usr/bin/ffmpeg")
    parser.add_argument("--ffprobe", default="/usr/bin/ffprobe")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Probe existing outputs before skipping them.",
    )
    return parser.parse_args()


def hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def mirror_tree(src: Path, dst: Path, *, hardlink: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            if hardlink:
                hardlink_or_copy(item, target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)


def update_info_json(dst: Path, *, height: int, width: int) -> None:
    info_path = dst / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    for key in VIDEO_KEYS:
        feature = info["features"][key]
        feature["shape"] = [height, width, 3]
        video_info = feature.setdefault("info", {})
        video_info["video.fps"] = float(info.get("fps", 30))
        video_info["video.height"] = height
        video_info["video.width"] = width
        video_info["video.channels"] = 3
        video_info["video.codec"] = "h264"
        video_info["video.pix_fmt"] = "yuv420p"
        video_info["video.is_depth_map"] = False
        video_info["has_audio"] = False

    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
        f.write("\n")


def probe_ok(path: Path, *, ffprobe: str, height: int, width: int) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        streams = json.loads(result.stdout).get("streams", [])
    except Exception:
        return False
    if not streams:
        return False
    stream = streams[0]
    return (
        stream.get("codec_name") == "h264"
        and int(stream.get("width", -1)) == width
        and int(stream.get("height", -1)) == height
        and stream.get("avg_frame_rate") in {"30/1", "30/1/0"}
    )


def convert_one(task: tuple[Path, Path], args: argparse.Namespace) -> tuple[str, str]:
    src, dst = task
    if args.verify_existing:
        if probe_ok(dst, ffprobe=args.ffprobe, height=args.height, width=args.width):
            return "skipped", str(dst)
    elif dst.exists() and dst.stat().st_size > 0:
        return "skipped", str(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + f".tmp-{os.getpid()}")
    if tmp.exists():
        tmp.unlink()

    cmd = [
        args.ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        "1",
        "-i",
        str(src),
        "-vf",
        f"scale={args.width}:{args.height}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        args.preset,
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        tmp.replace(dst)
        return "converted", str(dst)
    except subprocess.CalledProcessError as exc:
        if tmp.exists():
            tmp.unlink()
        msg = (exc.stderr or exc.stdout or "").strip().splitlines()[-5:]
        return "failed", f"{src}: {' | '.join(msg)}"


def main() -> int:
    args = parse_args()
    src = args.src.resolve()
    dst = args.dst.resolve()

    if not (src / "videos").exists():
        raise FileNotFoundError(src / "videos")

    print(f"[setup] src={src}")
    print(f"[setup] dst={dst}")
    print(f"[setup] target={args.width}x{args.height} codec=h264 fps=30")

    (dst / "videos").mkdir(parents=True, exist_ok=True)
    print("[setup] mirroring data with hardlinks")
    mirror_tree(src / "data", dst / "data", hardlink=True)
    print("[setup] copying meta")
    mirror_tree(src / "meta", dst / "meta", hardlink=False)
    update_info_json(dst, height=args.height, width=args.width)

    tasks: list[tuple[Path, Path]] = []
    for video in sorted((src / "videos").rglob("*.mp4")):
        rel = video.relative_to(src / "videos")
        tasks.append((video, dst / "videos" / rel))
    if args.limit > 0:
        tasks = tasks[: args.limit]
    print(f"[convert] videos={len(tasks)} workers={args.workers}")

    converted = skipped = failed = 0
    start = time.time()
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(convert_one, task, args) for task in tasks]
        for idx, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            status, detail = fut.result()
            if status == "converted":
                converted += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                failures.append(detail)

            if idx == 1 or idx % 500 == 0 or idx == len(tasks):
                elapsed = max(time.time() - start, 1e-6)
                rate = idx / elapsed
                remaining = (len(tasks) - idx) / rate if rate > 0 else 0
                print(
                    "[progress] "
                    f"{idx}/{len(tasks)} converted={converted} skipped={skipped} "
                    f"failed={failed} rate={rate:.2f}/s eta={remaining/60:.1f}m",
                    flush=True,
                )

    if failures:
        fail_path = dst / "resize_failures.txt"
        fail_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        print(f"[done] failures={failed}; see {fail_path}", file=sys.stderr)
        return 1

    print(f"[done] converted={converted} skipped={skipped} failed={failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
