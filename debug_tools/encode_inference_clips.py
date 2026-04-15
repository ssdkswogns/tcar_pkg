#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class ClipSegment:
    folder: Path
    image_paths: list[Path]

    @property
    def start_frame(self) -> int:
        return parse_frame_index(self.image_paths[0])

    @property
    def end_frame(self) -> int:
        return parse_frame_index(self.image_paths[-1])


def parse_frame_index(image_path: Path) -> int:
    stem_parts = image_path.stem.split("_")
    if not stem_parts or not stem_parts[0].isdigit():
        raise ValueError(f"Could not parse frame index from filename: {image_path.name}")
    return int(stem_parts[0])


def collect_image_folders(input_root: Path, source_dir_name: str) -> list[Path]:
    if collect_images(input_root):
        return [input_root]

    if input_root.name == source_dir_name and collect_images(input_root):
        return [input_root]

    folders = sorted(input_root.glob(f"traffic_260323_*/{source_dir_name}"))
    if not folders:
        folders = sorted(p for p in input_root.rglob(source_dir_name) if p.is_dir())
    return [folder for folder in folders if collect_images(folder)]


def resolve_input_root(input_root: Path) -> Path:
    input_root = Path(input_root)
    if (input_root / "vis").is_dir():
        return input_root / "vis"
    return input_root


def collect_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def split_into_clips(image_paths: list[Path], max_frame_gap: int) -> list[ClipSegment]:
    if not image_paths:
        return []

    clips: list[ClipSegment] = []
    current_clip = [image_paths[0]]
    previous_frame = parse_frame_index(image_paths[0])

    for image_path in image_paths[1:]:
        current_frame = parse_frame_index(image_path)
        if current_frame - previous_frame <= max_frame_gap:
            current_clip.append(image_path)
        else:
            clips.append(ClipSegment(folder=image_paths[0].parent, image_paths=current_clip))
            current_clip = [image_path]
        previous_frame = current_frame

    clips.append(ClipSegment(folder=image_paths[0].parent, image_paths=current_clip))
    return clips


def ensure_output_dir(folder: Path, input_root: Path, output_root: Path) -> Path:
    rel_dir = folder.relative_to(input_root)
    out_dir = output_root / rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_clip_output_path(
    clip: ClipSegment,
    input_root: Path,
    output_root: Path,
    clip_index: int,
) -> Path:
    out_dir = ensure_output_dir(clip.folder, input_root, output_root)
    filename = (
        f"clip_{clip_index:04d}_{clip.start_frame:06d}_{clip.end_frame:06d}_"
        f"{len(clip.image_paths):04d}f.mp4"
    )
    return out_dir / filename


def encode_clip(
    clip: ClipSegment,
    output_path: Path,
    fps: float,
    codec: str,
) -> None:
    first_frame = cv2.imread(str(clip.image_paths[0]), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError(f"Failed to read first frame: {clip.image_paths[0]}")

    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    try:
        writer.write(first_frame)
        for image_path in clip.image_paths[1:]:
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Failed to read frame: {image_path}")
            if frame.shape[:2] != (height, width):
                raise RuntimeError(
                    f"Frame size mismatch: {image_path} has {frame.shape[:2]}, expected {(height, width)}"
                )
            writer.write(frame)
    finally:
        writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split consecutive inference images into MP4 clips."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(__file__).resolve().parent / "infer",
        help="Inference output root or vis root containing saved images.",
    )
    parser.add_argument(
        "--source-dir-name",
        type=str,
        default="camera_3__compressed",
        help="Image folder name to search under each sequence directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root directory for encoded clips. Defaults to <input-root>_clips.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output clip fps.",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="OpenCV fourcc codec for output videos.",
    )
    parser.add_argument(
        "--max-frame-gap",
        type=int,
        default=1,
        help="Maximum allowed frame index gap inside one clip.",
    )
    parser.add_argument(
        "--min-clip-frames",
        type=int,
        default=1,
        help="Minimum number of frames required to emit a clip.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip clips whose output file already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.input_root = resolve_input_root(args.input_root)

    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")
    if args.output_root is None:
        args.output_root = args.input_root.parent / f"{args.input_root.name}_clips"
    if len(args.codec) != 4:
        raise ValueError(f"Codec must be a fourcc string of length 4: {args.codec}")
    if args.max_frame_gap < 1:
        raise ValueError("--max-frame-gap must be >= 1")
    if args.min_clip_frames < 1:
        raise ValueError("--min-clip-frames must be >= 1")

    folders = collect_image_folders(args.input_root, args.source_dir_name)
    if not folders:
        raise FileNotFoundError(
            f"No folders named {args.source_dir_name} found under {args.input_root}"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)

    print(f"Input root: {args.input_root}")
    print(f"Output root: {args.output_root}")
    print(f"FPS: {args.fps}")
    print(f"Codec: {args.codec}")
    print(f"Max frame gap: {args.max_frame_gap}")
    print(f"Min clip frames: {args.min_clip_frames}")

    encoded_count = 0
    skipped_existing_count = 0
    total_clip_candidates = 0

    for folder in folders:
        image_paths = collect_images(folder)
        clips = split_into_clips(image_paths, args.max_frame_gap)

        for clip_index, clip in enumerate(clips, start=1):
            if len(clip.image_paths) < args.min_clip_frames:
                continue

            total_clip_candidates += 1
            output_path = build_clip_output_path(
                clip=clip,
                input_root=args.input_root,
                output_root=args.output_root,
                clip_index=clip_index,
            )

            if args.skip_existing and output_path.exists():
                skipped_existing_count += 1
                continue

            encode_clip(
                clip=clip,
                output_path=output_path,
                fps=args.fps,
                codec=args.codec,
            )
            encoded_count += 1
            print(
                f"[{encoded_count}] {output_path.name} "
                f"({clip.start_frame} -> {clip.end_frame}, {len(clip.image_paths)} frames)"
            )

    print("Done.")
    print(f"Clip candidates: {total_clip_candidates}")
    print(f"Encoded clips: {encoded_count}")
    print(f"Skipped existing clips: {skipped_existing_count}")


if __name__ == "__main__":
    main()
