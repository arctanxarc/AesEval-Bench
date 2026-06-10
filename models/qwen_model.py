"""
Adapter for local Qwen2.5-VL checkpoints.

Supported checkpoints:
  - Qwen2.5-VL-3B-Instruct
  - Qwen2.5-VL-7B-Instruct
  - Qwen2.5-VL-32B-Instruct
  - Qwen2.5-VL-72B-Instruct
  - Additional Qwen2.5-VL variants via ``model_path``

Multi-GPU strategy:
  ``device_map="auto"`` lets accelerate shard the model across visible GPUs.
  A single device can also be selected explicitly through ``device_id``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from models.base_model import BaseModel

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

logger = logging.getLogger(__name__)


class QwenVLModel(BaseModel):
    """Qwen2.5-VL multimodal model with single- or multi-GPU loading."""

    # Mapping from model type to Hugging Face repo ID.
    MODEL_REPO_MAP: Dict[str, str] = {
        "qwen2.5-vl-3b":  "Qwen/Qwen2.5-VL-3B-Instruct",
        "qwen2.5-vl-7b":  "Qwen/Qwen2.5-VL-7B-Instruct",
        "qwen2.5-vl-32b": "Qwen/Qwen2.5-VL-32B-Instruct",
        "qwen2.5-vl-72b": "Qwen/Qwen2.5-VL-72B-Instruct",
    }

    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        super().__init__(model_config)
        cfg = model_config or {}

        # Resolve the checkpoint path or repo ID.
        model_type = cfg.get("model_type", "qwen2.5-vl-7b").lower()
        self.repo_id: str = cfg.get(
            "model_path",
            self.MODEL_REPO_MAP.get(model_type, model_type),
        )

        # Device placement strategy
        device_ids = cfg.get("device_ids", None)       # e.g. [0,1,2,3]
        device_id = cfg.get("device_id", None)         # e.g. 0

        if device_ids is not None and len(device_ids) > 1:
            # Multi-GPU mode: let accelerate shard weights automatically.
            self.device_map = "auto"
            # Limit visible devices to the requested GPU subset.
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in device_ids)
        elif device_id is not None:
            self.device_map = {"": f"cuda:{device_id}"}
        else:
            self.device_map = "auto"

        self.processor = None
        self.model = None

    # Initialization
    def load_model(self) -> None:
        if self.processor is not None and self.model is not None:
            return

        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        logger.info("Loading Qwen2.5-VL: %s (device_map=%s)", self.repo_id, self.device_map)

        self.processor = AutoProcessor.from_pretrained(
            self.repo_id,
            trust_remote_code=True,
        )

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.repo_id,
            torch_dtype=torch.float16,
            device_map=self.device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()

        logger.info("Qwen2.5-VL loaded successfully.")

    # Inference
    def predict(self, input_data: Dict[str, Any]) -> str:
        import torch
        from PIL import Image

        question = input_data.get("question", "").strip()
        if not question:
            raise ValueError("`question` is required")

        image_path = input_data.get("image_path")

        if self.model is None:
            self.load_model()

        # Append meta_info for the text-preview family of tasks.
        eval_mode = input_data.get("eval_mode", "")
        if "text-preview" in eval_mode:
            meta_info = input_data.get("meta_info", "")
            question = (
                f"The simplified original metainfos are {meta_info}, "
                f'the question is "{question}"'
            )

        # Build chat-style messages.
        content: list = []
        if image_path and os.path.exists(str(image_path)):
            content.append({"type": "image", "image": str(image_path)})
        content.append({"type": "text", "text": question})

        messages = [{"role": "user", "content": content}]

        text_prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Preprocess text and optional image inputs.
        if image_path and os.path.exists(str(image_path)):
            image = Image.open(str(image_path)).convert("RGB")
            inputs = self.processor(
                text=[text_prompt],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=[text_prompt],
                padding=True,
                return_tensors="pt",
            )

        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        # Remove the prompt tokens and decode only newly generated tokens.
        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.tokenizer.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return output.rstrip(".")
