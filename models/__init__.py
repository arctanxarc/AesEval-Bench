"""
Model registry and factory helpers.

Supported model families:
  - GPT-style models accessed through an OpenAI-compatible API
  - Local Qwen2.5-VL checkpoints
"""

from typing import Any, Dict, Optional, Type

from models.base_model import BaseModel
from models.gpt_model import GPTModel
from models.qwen_model import QwenVLModel

# Model registry
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    # GPT-style API models
    "gpt-4o":        GPTModel,
    "gpt-4o-mini":   GPTModel,
    "gpt-4.1":       GPTModel,
    "gpt-4.1-mini":  GPTModel,
    "gpt-4.1-nano":  GPTModel,
    "o3":       GPTModel,
    "gpt-5": GPTModel,
    # Local Qwen2.5-VL models
    "qwen2.5-vl-3b":  QwenVLModel,
    "qwen2.5-vl-7b":  QwenVLModel,
    "qwen2.5-vl-32b": QwenVLModel,
    "qwen2.5-vl-72b": QwenVLModel,
}


def is_api_model(model_type: str) -> bool:
    """Return whether a model type is served through an API."""
    cls = MODEL_REGISTRY.get(model_type)
    return cls is GPTModel


def get_model(
    model_type: str,
    model_config: Optional[Dict[str, Any]] = None,
) -> BaseModel:
    """
    Create a model instance from the registry.

    Args:
        model_type: Registry key such as ``"gpt-4o-mini"`` or ``"qwen2.5-vl-7b"``.
        model_config: Configuration dictionary passed to the model constructor.

    Returns:
        Instantiated model object. The model is created but not necessarily loaded.

    Raises:
        ValueError: The model type is not registered.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    model_class = MODEL_REGISTRY[model_type]
    cfg = (model_config or {}).copy()
    cfg.setdefault("model_type", model_type)

    return model_class(cfg)
