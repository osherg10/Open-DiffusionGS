import argparse
import json
import importlib
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute FID and KID between two folders of images.")
    parser.add_argument("--reference", required=True, help="Directory of reference images.")
    parser.add_argument("--generated", required=True, help="Directory of generated images.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for metric computation.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device to run metrics on (defaults to CUDA if available).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=299,
        help="Resize images to this square size before feeding to the inception model.",
    )
    parser.add_argument(
        "--kid-subset-size",
        type=int,
        default=100,
        help="Subset size used internally for KID estimation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON file to write the computed metrics to.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compute_fid_kid = importlib.import_module("diffusionGS.eval.metrics").compute_fid_kid
    metrics = compute_fid_kid(
        reference_dir=args.reference,
        generated_dir=args.generated,
        batch_size=args.batch_size,
        device=args.device,
        image_size=args.image_size,
        kid_subset_size=args.kid_subset_size,
        save_path=args.output,
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
