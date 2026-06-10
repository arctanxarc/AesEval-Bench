"""
Base model interface shared by all model adapters.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseModel(ABC):
    """Unified interface for multimodal models."""

    @abstractmethod
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """
        Args:
            model_config: Model configuration, such as model type, API key,
                or device settings.
        """

    @abstractmethod
    def load_model(self) -> None:
        """Load or initialize the model backend."""

    @abstractmethod
    def predict(self, input_data: Dict[str, Any]) -> str:
        """
        Run inference for a single input.

        Args:
            input_data: Input dictionary containing at least ``question`` and
                ``image_path``, and optionally ``eval_mode`` and ``meta_info``.

        Returns:
            Model-generated text output.
        """
