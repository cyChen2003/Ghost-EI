from __future__ import annotations

import argparse

from cli_utils import add_remote_inference_args, extract_remote_kwargs
from unified_agent import OSAtlasAgent, run_all_tests
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ""
DEFAULT_BASE_DATA_PATH = str(REPO_ROOT / "datasets" / "GhostEI.jsonl")
DEFAULT_LOG_DIR = str((Path(__file__).resolve().parent / "outputs").resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OS-Atlas agent with the unified MobileAgent action space.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--processor-path", default=None, help="Defaults to model path when omitted.")
    parser.add_argument("--base-data-path", default=DEFAULT_BASE_DATA_PATH)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    add_remote_inference_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processor_path = args.processor_path or args.model_path
    remote_kwargs = extract_remote_kwargs(args)
    agent = OSAtlasAgent(
        model_path=args.model_path,
        processor_path=processor_path,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
        **remote_kwargs,
    )
    summary = run_all_tests(agent, base_data_path=args.base_data_path, log_dir=args.log_dir)
    print(summary)


if __name__ == "__main__":
    main()
