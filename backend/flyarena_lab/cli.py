import argparse
import json
from pathlib import Path
from .experiment import Options, run, reevaluate


def main():
    parser = argparse.ArgumentParser(description="Authored circuit training and intervention evidence")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--reevaluate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.reevaluate:
        if args.reevaluate.stat().st_size > 16 * 1024**2:
            parser.error("Export exceeds 16 MiB")
        result = reevaluate(json.loads(args.reevaluate.read_text()), args.device)
    else:
        settings = dict(population=4, generations=1, ticks=30, training_seeds=4, heldout_seeds=8) if args.quick else {}
        result = run(Options(device=args.device, seed=args.seed, **settings))
    args.output.write_text(json.dumps(result, allow_nan=False))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
