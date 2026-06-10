"""
Base task class defining the shared prepare_data / format_input / evaluate flow.

Subclasses only need to define task-specific prompt attributes such as
``self.balance_prompt``. Data loading, input formatting, and scoring are handled
here.
"""

import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)


class BaseTask:
    """Base class for graphic design aesthetics evaluation tasks."""

    # Shared prompt fragments
    QA_SUFFIX = "Answer it with one word 'yes' or 'no'."

    BBOX_SUFFIX = (
        "Return the bounding box of the Not aesthetic element in the format of "
        "'left,top,width,height'. Norm to 0~1. Precise to decimal as possible "
        "and make sure only output the bounding box. "
        "If there is no any problems, please return 'None'."
    )

    CHOICE_PREFIX = (
        "There may exist some region that is not aesthetic given the preview "
        "and question. Please choose the most likely one from the options. "
        "If there is no any problems, you can choose the option of 'None'."
    )

    def __init__(self, task_config: Optional[Dict[str, Any]] = None):
        cfg = task_config or {}
        self.dimension: str = cfg.get("dimension", "")
        self.task_name: str = cfg.get("task_name", "")

    # Data loading
    def prepare_data(self, data_path: Path) -> Dict[str, Any]:
        """Load metadata, labels, and preview assets from a sample directory."""
        with open(data_path / "simplified_meta_info.json", "r") as f:
            meta_info = json.load(f)

        with open(data_path / "GT.json", "r") as f:
            gt = json.load(f)

        preview_path = data_path / "preview.png"

        return {
            "meta_info": meta_info,
            "preview_path": preview_path,
            "original_path": preview_path,
            "changes": gt,
            "task_type": self.task_name,
        }

    # Prompt lookup
    def _get_task_prompt(self) -> str:
        """Resolve the task-specific prompt defined by the subclass."""
        attr = f"{self.task_name.lower()}_prompt"
        prompt = getattr(self, attr, None)
        if prompt is None:
            raise AttributeError(
                f"{self.__class__.__name__} has no prompt attribute '{attr}'"
            )
        return prompt

    # GT key helper
    def _gt_key(self) -> str:
        return f"{self.dimension}-{self.task_name}"

    # Input formatting
    def format_input(
        self,
        data: Dict[str, Any],
        eval_mode: str = "text-preview",
        model_type: str = "",
    ) -> Dict[str, Any]:
        """
        Build the model input for the requested evaluation mode.

        Returns:
            Dictionary containing fields such as question, image_path,
            ground_truth, eval_mode, and meta_info.
        """
        gt_key = self._gt_key()
        changes = data["changes"].get(gt_key, [])
        has_issue = bool(changes)
        task_prompt = self._get_task_prompt()

        # Use the preview image when an issue exists; otherwise fall back to
        # the original image path.
        image_path = data["preview_path"] if has_issue else data["original_path"]

        if eval_mode == "text-preview":
            return self._format_yes_no(data, task_prompt, image_path, has_issue)
        elif eval_mode == "text-preview-choice":
            return self._format_choice(data, task_prompt, image_path, changes)
        elif eval_mode == "text-preview-bbox":
            return self._format_bbox(data, task_prompt, image_path, changes)
        else:
            raise ValueError(f"Unsupported eval_mode: {eval_mode}")

    def _format_yes_no(
        self,
        data: Dict[str, Any],
        task_prompt: str,
        image_path: Path,
        has_issue: bool,
    ) -> Dict[str, Any]:
        """Format input for binary yes/no evaluation."""
        return {
            "eval_mode": "text-preview",
            "task_type": data["task_type"],
            "meta_info": data["meta_info"],
            "image_path": str(image_path),
            "question": f"{task_prompt} {self.QA_SUFFIX}",
            "choices": None,
            "ground_truth": "yes" if has_issue else "no",
        }

    def _format_choice(
        self,
        data: Dict[str, Any],
        task_prompt: str,
        image_path: Path,
        changes: List[Dict],
    ) -> Dict[str, Any]:
        """Format input for four-way multiple-choice evaluation."""
        all_elements = data["meta_info"]["elements"]
        option_labels = ["A", "B", "C", "D"]

        gt_idx = self._extract_element_index(changes)

        if gt_idx is None:
            # No issue: sample 3 elements and add a None option.
            sampled = random.sample(all_elements, min(3, len(all_elements)))
            choices: List[Any] = [
                self._element_to_bbox(e) for e in sampled
            ]
            choices.append("None")
            random.shuffle(choices)
            ground_truth = option_labels[choices.index("None")]
        else:
            # Positive sample: include the GT box plus 3 distractors.
            gt_element = self._find_element(all_elements, gt_idx)
            if gt_element is None:
                return self._fallback_choice(data, task_prompt, image_path)

            gt_bbox = self._element_to_bbox(gt_element)
            others = [e for e in all_elements if e["element_index"] != gt_idx]
            sampled = random.sample(others, min(3, len(others)))
            choices = [gt_bbox] + [self._element_to_bbox(e) for e in sampled]
            random.shuffle(choices)
            ground_truth = option_labels[choices.index(gt_bbox)]

        # Build the option block shown to the model.
        choice_text = "\n"
        for i, c in enumerate(choices):
            if c == "None":
                choice_text += f"{option_labels[i]}: None\n"
            else:
                choice_text += f"{option_labels[i]}: [{c[0]}, {c[1]}, {c[2]}, {c[3]}]\n"

        question = (
            f"{self.CHOICE_PREFIX} {task_prompt}\n{choice_text}"
            "Please answer with only the letter (A, B, C, or D)."
        )

        return {
            "eval_mode": "text-preview-choice",
            "task_type": data["task_type"],
            "meta_info": data["meta_info"],
            "image_path": str(image_path),
            "question": question,
            "choices": choice_text,
            "ground_truth": ground_truth,
        }

    def _format_bbox(
        self,
        data: Dict[str, Any],
        task_prompt: str,
        image_path: Path,
        changes: List[Dict],
    ) -> Dict[str, Any]:
        """Format input for bounding-box prediction."""
        gt_idx = self._extract_element_index(changes)

        if gt_idx is None:
            ground_truth = "None"
        else:
            all_elements = data["meta_info"]["elements"]
            gt_element = self._find_element(all_elements, gt_idx)
            if gt_element is not None:
                ground_truth = self._element_to_bbox(gt_element)
            else:
                ground_truth = "None"

        question = f"{task_prompt} {self.BBOX_SUFFIX}"

        return {
            "eval_mode": "text-preview-bbox",
            "task_type": data["task_type"],
            "meta_info": data["meta_info"],
            "image_path": str(image_path),
            "question": question,
            "choices": None,
            "ground_truth": ground_truth,
        }

    # Scoring
    def evaluate(
        self,
        predictions: Dict[str, Any],
        eval_mode: str = "text-preview",
        model_type: str = "",
    ) -> float:
        """
        Score one model prediction.

        Returns:
            - text-preview: 1.0 (correct) / 0.0 (incorrect)
            - text-preview-choice: 1.0 / 0.0
            - text-preview-bbox: IoU in [0, 1]
        """
        pred = predictions["prediction"]
        if isinstance(pred, str) and "<think>" in pred and "</think>" in pred:
            pred_without_think = pred.split("</think>")[-1].strip()
        else:
            pred_without_think = pred
        predictions["prediction"] = pred_without_think
        if "bbox" in eval_mode:
            return self._evaluate_bbox(predictions)
        elif "choice" in eval_mode:
            return self._evaluate_choice(predictions)
        else:
            return self._evaluate_yes_no(predictions)

    def _evaluate_yes_no(self, predictions: Dict[str, Any]) -> float:
        pred = predictions["prediction"]
        gt = predictions["ground_truth"]
        if pred is None or pred == "None":
            return 0.0
        pred_lower = pred.lower()
        if "yes" in pred_lower:
            pred_norm = "yes"
        elif "no" in pred_lower:
            pred_norm = "no"
        else:
            pred_norm = pred
        return 1.0 if pred_norm == gt else 0.0

    def _evaluate_choice(self, predictions: Dict[str, Any]) -> float:
        pred = (predictions["prediction"] or "").strip().upper()
        gt = predictions["ground_truth"].strip().upper()
        # Extract the first valid option letter.
        pred_letter = ""
        for ch in pred:
            if ch in "ABCD":
                pred_letter = ch
                break
        return 1.0 if pred_letter == gt else 0.0

    def _evaluate_bbox(self, predictions: Dict[str, Any]) -> float:
        gt = predictions["ground_truth"]
        pred = predictions["prediction"]
        image_path = predictions.get("image_path")

        # Handle no-issue cases explicitly.
        if gt == "None":
            return 1.0 if (pred is None or "none" in str(pred).lower()) else 0.0
        if pred is None or "none" in str(pred).lower():
            return 0.0

        # Parse the ground-truth bbox.
        gt_bbox = [float(x) for x in gt]

        # Normalize GT pixel coordinates into [0, 1].
        if image_path and Path(image_path).exists():
            try:
                with Image.open(image_path) as img:
                    w, h = img.size
                    gt_bbox = [
                        gt_bbox[0] / w,
                        gt_bbox[1] / h,
                        gt_bbox[2] / w,
                        gt_bbox[3] / h,
                    ]
            except Exception as e:
                logger.warning("Failed to open image for normalization: %s", e)

        # Extract numeric coordinates from the model response.
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(pred))
        if len(nums) < 4:
            logger.warning("Invalid bbox prediction: %s", pred)
            return 0.0
        pred_bbox = [float(x) for x in nums[:4]]

        # Compute IoU where both boxes use [left, top, width, height].
        return self._compute_iou(gt_bbox, pred_bbox)

    @staticmethod
    def _compute_iou(
        box_a: List[float], box_b: List[float]
    ) -> float:
        """Compute IoU for two [left, top, width, height] boxes."""
        x_left = max(box_a[0], box_b[0])
        y_top = max(box_a[1], box_b[1])
        x_right = min(box_a[0] + box_a[2], box_b[0] + box_b[2])
        y_bottom = min(box_a[1] + box_a[3], box_b[1] + box_b[3])

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        inter = (x_right - x_left) * (y_bottom - y_top)
        area_a = box_a[2] * box_a[3]
        area_b = box_b[2] * box_b[3]
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    # Helper methods
    @staticmethod
    def _extract_element_index(changes: List[Dict]) -> Optional[int]:
        """Extract the first element_index from GT changes."""
        if not changes:
            return None
        idx_field = changes[0].get("element_index")
        if idx_field is None:
            return None
        # element_index may be stored as either a list or an int.
        if isinstance(idx_field, list):
            return idx_field[0] if idx_field else None
        return int(idx_field)

    @staticmethod
    def _find_element(
        elements: List[Dict], element_index: int
    ) -> Optional[Dict]:
        """Find an element by element_index in the element list."""
        for e in elements:
            if e["element_index"] == element_index:
                return e
        return None

    @staticmethod
    def _element_to_bbox(element: Dict) -> List[float]:
        """Extract [left, top, width, height] from an element record."""
        return [
            element["left"],
            element["top"],
            element["width"],
            element["height"],
        ]

    def _fallback_choice(
        self, data: Dict[str, Any], task_prompt: str, image_path: Path
    ) -> Dict[str, Any]:
        """Fallback for missing GT element metadata in choice mode."""
        return {
            "eval_mode": "text-preview-choice",
            "task_type": data["task_type"],
            "meta_info": data["meta_info"],
            "image_path": str(image_path),
            "question": f"{self.CHOICE_PREFIX} {task_prompt}",
            "choices": "",
            "ground_truth": "None",
        }
