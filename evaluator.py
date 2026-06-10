import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from tqdm import tqdm

from models import get_model, is_api_model
from models.base_model import BaseModel
from tasks import TASK_REGISTRY
from tasks.base_task import BaseTask

logger = logging.getLogger(__name__)


class Evaluator:
    """Benchmark evaluator."""

    def __init__(
        self,
        model_type: str,
        model_config: Dict[str, Any],
        benchmark_path: Path,
        eval_mode: str,
        max_workers: int = 4,
    ):
        self.model_type = model_type
        self.model_config = model_config
        self.benchmark_path = Path(benchmark_path)
        self.eval_mode = eval_mode
        self.max_workers = max_workers

    # Main entry point
    def evaluate(self) -> Dict[str, Any]:
        """Run evaluation for the full benchmark and return the result dictionary."""
        data_dirs = sorted(
            d for d in self.benchmark_path.iterdir()
            if d.is_dir() and "_new" in d.name
        )
        logger.info(
            "Found %d sample directories in %s", len(data_dirs), self.benchmark_path
        )

        # Create and load the model.
        model = get_model(self.model_type, self.model_config)
        model.load_model()

        # Choose the execution strategy based on the model type.
        if is_api_model(self.model_type):
            all_results = self._evaluate_concurrent(model, data_dirs)
        else:
            all_results = self._evaluate_sequential(model, data_dirs)

        return self._compute_final_results(all_results)

    # Concurrent execution for API-based models
    def _evaluate_concurrent(
        self, model: BaseModel, data_dirs: List[Path]
    ) -> List[Dict]:
        """Evaluate with a thread pool for API-based models."""
        import threading

        all_results: List[Dict] = []
        failed_count = 0
        lock = threading.Lock()
        pbar = tqdm(total=len(data_dirs), desc="Evaluating")

        def _worker(data_dir: Path) -> None:
            nonlocal failed_count
            try:
                result = self._evaluate_single_dir(model, data_dir)
                if result is not None:
                    with lock:
                        all_results.append(result)
            except Exception as e:
                with lock:
                    failed_count += 1
                logger.error(
                    "Unexpected error evaluating %s: %s", data_dir.name, e
                )
            finally:
                pbar.update(1)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(_worker, d) for d in data_dirs]
            # Wait for all submitted jobs to finish.
            for f in futures:
                f.result()

        pbar.close()

        if failed_count > 0:
            logger.warning(
                "%d sample(s) failed due to unexpected errors.", failed_count
            )

        return all_results

    # Sequential execution for local models
    def _evaluate_sequential(
        self, model: BaseModel, data_dirs: List[Path]
    ) -> List[Dict]:
        """Run sequential inference for local models."""
        all_results: List[Dict] = []

        for data_dir in tqdm(data_dirs, desc="Evaluating"):
            try:
                result = self._evaluate_single_dir(model, data_dir)
                if result is not None:
                    all_results.append(result)
            except Exception as e:
                logger.error(
                    "Unexpected error evaluating %s: %s", data_dir.name, e
                )

        return all_results

    # Single-sample evaluation
    def _evaluate_single_dir(
        self, model: BaseModel, data_dir: Path
    ) -> Optional[Dict]:
        """Evaluate one sample directory and return all task-level results."""
        gt_path = data_dir / "GT.json"
        if not gt_path.exists():
            return None

        with open(gt_path, "r") as f:
            gt_json = json.load(f)

        sample_name = data_dir.name
        results: Dict[str, Dict[str, Dict[str, float]]] = {}
        sample_metrics = {"total": 0, "correct": 0.0}
        prediction_details: List[Dict] = []

        for gt_key in gt_json:
            parts = gt_key.split("-")
            if len(parts) != 2:
                continue
            dimension, task = parts

            task_class = TASK_REGISTRY.get(dimension)
            if task_class is None:
                continue

            task_instance = task_class({
                "dimension": dimension,
                "task_name": task,
            })

            # Prepare data and format the model input.
            data = task_instance.prepare_data(data_dir)
            formatted = task_instance.format_input(
                data, self.eval_mode, self.model_type
            )

            # Model inference. API models may return None on failure.
            prediction = model.predict(formatted)

            # Score the prediction.
            pred_dict = {
                "prediction": prediction,
                "ground_truth": formatted["ground_truth"],
                "image_path": formatted["image_path"],
            }
            score = task_instance.evaluate(pred_dict, self.eval_mode, self.model_type)

            # Mark API failures explicitly when prediction is None.
            is_failed = prediction is None

            # Record per-task metrics.
            results.setdefault(dimension, {})
            results[dimension].setdefault(task, {"total": 0, "correct": 0.0, "failed": 0})
            results[dimension][task]["total"] += 1
            results[dimension][task]["correct"] += score
            if is_failed:
                results[dimension][task]["failed"] += 1

            sample_metrics["total"] += 1
            sample_metrics["correct"] += score

            prediction_details.append({
                "dimension": dimension,
                "task": task,
                "prediction": {
                    "prediction": prediction,
                    "ground_truth": formatted["ground_truth"],
                },
                "choices": formatted.get("choices"),
                "result": score,
                "api_failed": is_failed,
            })

        return {
            "sample_name": sample_name,
            "results": results,
            "sample_metrics": sample_metrics,
            "prediction_details": prediction_details,
        }

    # Result aggregation
    def _compute_final_results(
        self, all_results: List[Dict]
    ) -> Dict[str, Any]:
        """Aggregate per-sample results into final benchmark metrics."""
        agg_results: Dict[str, Dict[str, Dict[str, float]]] = {}
        sample_results: Dict[str, Dict[str, float]] = {}
        all_details: Dict[str, List[Dict]] = {}
        total_failed = 0

        for item in all_results:
            name = item["sample_name"]
            sample_results[name] = item["sample_metrics"]
            all_details[name] = item["prediction_details"]

            for dim, tasks in item["results"].items():
                agg_results.setdefault(dim, {})
                for task, metrics in tasks.items():
                    agg_results[dim].setdefault(
                        task, {"total": 0, "correct": 0.0, "failed": 0}
                    )
                    agg_results[dim][task]["total"] += metrics["total"]
                    agg_results[dim][task]["correct"] += metrics["correct"]
                    agg_results[dim][task]["failed"] += metrics.get("failed", 0)
                    total_failed += metrics.get("failed", 0)

        # Compute per-task and per-dimension accuracy.
        final: Dict[str, Any] = {}
        overall_acc = 0.0
        dim_count = 0

        for dimension, tasks in agg_results.items():
            final[dimension] = {}
            dim_acc = 0.0
            task_count = 0

            for task, m in tasks.items():
                acc = m["correct"] / m["total"] if m["total"] > 0 else 0.0
                task_result = {"accuracy": acc, "total": m["total"]}
                if m["failed"] > 0:
                    task_result["failed"] = m["failed"]
                final[dimension][task] = task_result
                dim_acc += acc
                task_count += 1

            avg = dim_acc / task_count if task_count > 0 else 0.0
            final[dimension]["average"] = {"accuracy": avg}
            overall_acc += avg
            dim_count += 1

        # Compute sample-level accuracy.
        sample_acc_sum = 0.0
        for m in sample_results.values():
            sample_acc_sum += (m["correct"] / m["total"]) if m["total"] > 0 else 0.0
        sample_level_acc = (
            sample_acc_sum / len(sample_results) if sample_results else 0.0
        )

        final["overall"] = {
            "accuracy": overall_acc / dim_count if dim_count > 0 else 0.0,
            "sample_level_accuracy": sample_level_acc,
        }

        # Record API failure statistics.
        if total_failed > 0:
            final["api_failures"] = total_failed
            logger.warning(
                "%d API call(s) failed (returned None), scored as 0.",
                total_failed,
            )

        # In bbox mode, also report mean IoU for samples with non-None GT.
        if "bbox" in self.eval_mode:
            total_iou, count = 0.0, 0
            for details in all_details.values():
                for d in details:
                    gt = d["prediction"]["ground_truth"]
                    if gt != "None":
                        total_iou += d["result"]
                        count += 1
            final["bbox_non_none_gt_iou_mean"] = total_iou / count if count > 0 else 0.0

        final["prediction_details"] = all_details
        return final

    # Output persistence
    @staticmethod
    def save_results(results: Dict[str, Any], output_path: Path) -> None:
        """Save the result dictionary as a JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        logger.info("Results saved to %s", output_path)
