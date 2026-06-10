import argparse
import logging
from pathlib import Path

from evaluator import Evaluator

# Default values
DEFAULT_MODEL_TYPE = "gpt-4o-mini"
DEFAULT_BENCHMARK_PATH = "./benchmark_data"
DEFAULT_OUTPUT_DIR = "./results"
DEFAULT_MAX_WORKERS = 4

SUPPORTED_EVAL_MODES = [
    "text-preview",          # Binary yes/no prediction
    "text-preview-choice",   # Four-way multiple choice (A/B/C/D)
    "text-preview-bbox",     # Bounding box prediction
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DesignBench evaluation tool"
    )

    # Model arguments
    parser.add_argument(
        "--model_type",
        type=str,
        default=DEFAULT_MODEL_TYPE,
        help="Model type, for example gpt-4o-mini or qwen2.5-vl-7b.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Local checkpoint path for Qwen models. Defaults to Hugging Face.",
    )

    # API arguments for OpenAI-compatible models
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key. Can also be provided through OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default=None,
        help="OpenAI-compatible base URL. Can also be set through OPENAI_BASE_URL.",
    )

    # Device arguments for local Qwen models
    parser.add_argument(
        "--device_ids",
        type=str,
        default=None,
        help="Comma-separated GPU device IDs, for example 0,1,2,3.",
    )

    # Evaluation arguments
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="text-preview",
        choices=SUPPORTED_EVAL_MODES,
        help="Evaluation mode.",
    )
    parser.add_argument(
        "--benchmark_path",
        type=str,
        default=DEFAULT_BENCHMARK_PATH,
        help="Path to the benchmark data directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used to save JSON results.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Number of concurrent workers for API-based models.",
    )

    return parser.parse_args()


def build_model_config(args: argparse.Namespace) -> dict:
    """Build a model configuration dictionary from CLI arguments."""
    cfg = {"model_type": args.model_type, "device": "cuda"}

    if args.model_path:
        cfg["model_path"] = args.model_path
    if args.api_key:
        cfg["api_key"] = args.api_key
    if args.api_base:
        cfg["api_base"] = args.api_base
    if args.device_ids:
        cfg["device_ids"] = [int(x) for x in args.device_ids.split(",")]

    return cfg


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    model_config = build_model_config(args)
    benchmark_path = Path(args.benchmark_path)

    logging.info("Model:     %s", args.model_type)
    logging.info("Eval mode: %s", args.eval_mode)
    logging.info("Benchmark: %s", benchmark_path)

    evaluator = Evaluator(
        model_type=args.model_type,
        model_config=model_config,
        benchmark_path=benchmark_path,
        eval_mode=args.eval_mode,
        max_workers=args.max_workers,
    )

    results = evaluator.evaluate()

    output_path = Path(args.output_dir) / f"{args.model_type}_{args.eval_mode}.json"
    evaluator.save_results(results, output_path)

    logging.info("Done! Results saved to %s", output_path)


if __name__ == "__main__":
    main()
