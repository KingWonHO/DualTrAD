"""Train and evaluate one model from a YAML configuration.

    python -m scripts.run --config config/qas/dualtrad.yaml --output runs/dualtrad
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.system import run_from_config
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a run configuration")
    parser.add_argument("--output", required=True, help="directory for checkpoints and results")
    parser.add_argument("--seed", type=int, default=None, help="override experiment.seed")
    parser.add_argument(
        "--dry-run", action="store_true", help="build everything and report shapes, then stop"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed

    if args.dry_run:
        import torch
        from src.system import build_model, build_objective

        model = build_model(config)
        context = torch.randn(2, config["data"]["context_length"], config["model"]["input_dim"])
        output = model(context)
        print(json.dumps({
            "model": model.get_config(),
            "objective": build_objective(config).get_config(),
            "prediction_shape": list(output.prediction.shape),
            "reconstruction_shape": (
                None if output.reconstruction is None else list(output.reconstruction.shape)
            ),
        }, indent=2, default=float))
        return

    payload = run_from_config(config, args.output)
    print(json.dumps(payload["metrics"], indent=2, default=float))


if __name__ == "__main__":
    main()
