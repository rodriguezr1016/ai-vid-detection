from __future__ import annotations

import argparse
import json

from .pipeline import VideoDetectionPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze whether a video is AI-generated.")
    parser.add_argument("video_path", help="Absolute or relative path to a video file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = VideoDetectionPipeline()
    result = pipeline.analyze(args.video_path)
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
