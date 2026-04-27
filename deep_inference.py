import argparse
import json
from pathlib import Path

from deep_vision import predict_image


def parse_args():
    parser = argparse.ArgumentParser(description="Run SortSmart deep checkpoint inference.")
    parser.add_argument("--checkpoint", type=Path, default=Path("output/deep/best_model.pt"))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    result = predict_image(args.checkpoint, args.image, device=args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
