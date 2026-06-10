"""
Adapter for GPT-style models served through an OpenAI-compatible API.

Supported examples include:
  gpt-4o, gpt-4o-mini, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, o3, gpt-5

Error handling strategy:
  - Retryable failures (429 / 500 / 502 / 503 / timeouts): exponential backoff
  - Non-retryable failures (400 / 401 / 403 ...): log and return None
"""

import base64
import io
import logging
import os
import time
from typing import Any, Dict, Optional

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)
from PIL import Image

from models.base_model import BaseModel

logger = logging.getLogger(__name__)

# Retryable HTTP status codes
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503}

# Retry parameters
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0   # Initial retry delay in seconds
RETRY_MAX_DELAY = 30.0    # Maximum retry delay in seconds


class GPTModel(BaseModel):
    """Model wrapper backed by the OpenAI Chat Completions API."""

    # Mapping from registry name to backend API model name.
    MODEL_NAME_MAP: Dict[str, str] = {
        "gpt-4o":        "gpt-4o",
        "gpt-4o-mini":   "gpt-4o-mini",
        "gpt-4.1":       "gpt-4.1",
        "gpt-4.1-mini":  "gpt-4.1-mini",
        "gpt-4.1-nano":  "gpt-4.1-nano",
        "o3":       "o3",
        "gpt-5":         "gpt-5",
    }

    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        super().__init__(model_config)
        cfg = model_config or {}

        # Resolve the backend model name.
        raw_name = cfg.get("model_type", "gpt-4o-mini")
        self.model_name: str = self.MODEL_NAME_MAP.get(raw_name, raw_name)

        # Prefer explicit config values and fall back to environment variables.
        self.api_key: str = cfg.get(
            "api_key",
            os.environ.get("OPENAI_API_KEY", ""),
        )
        self.api_base: str = cfg.get(
            "api_base",
            os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )

        self.system_prompt: str = cfg.get(
            "system_prompt", "You are a helpful assistant."
        )
        self.client: Optional[OpenAI] = None

    # Initialization
    def load_model(self) -> None:
        if self.client is not None:
            return
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            http_client=httpx.Client(
                base_url=self.api_base,
                follow_redirects=True,
            ),
        )
        logger.info("GPTModel initialized: %s @ %s", self.model_name, self.api_base)

    # Helper methods
    IMAGE_MAX_SIZE = 1024  # Maximum size of the longest image side in pixels

    @classmethod
    def _encode_image(cls, image_path: str) -> str:
        """Encode a local image as a base64 JPEG data URL."""
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Downscale images whose longer side exceeds the configured limit.
        w, h = image.size
        if max(w, h) > cls.IMAGE_MAX_SIZE:
            scale = cls.IMAGE_MAX_SIZE / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)
            logger.debug(
                "Resized image %s: (%d, %d) -> (%d, %d)",
                image_path, w, h, new_w, new_h,
            )

        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        return f"data:image/jpeg;base64,{b64}"

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Return whether an exception should be retried."""
        if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
            return True
        if isinstance(exc, APIStatusError) and exc.status_code in _RETRYABLE_STATUS_CODES:
            return True
        return False

    # Inference
    def predict(self, input_data: Dict[str, Any]) -> Optional[str]:
        """
        Run inference through the API backend.

        Returns:
            Model response text, or None if all retries fail or a
            non-retryable error occurs.
        """
        if self.client is None:
            self.load_model()

        user_content: list = []

        # Image input
        image_path = input_data.get("image_path", "")
        if image_path and os.path.exists(str(image_path)):
            b64_url = self._encode_image(str(image_path))
            user_content.append({
                "type": "image_url",
                "image_url": {"url": b64_url},
            })

        # Text prompt
        prompt_text = input_data.get("question", "")
        eval_mode = input_data.get("eval_mode", "")
        if "text-preview" in eval_mode:
            meta_info = input_data.get("meta_info", "")
            prompt_text = (
                f"The simplified original metainfos are {meta_info}, "
                f'the question is "{prompt_text}"'
            )
        user_content.append({"type": "text", "text": prompt_text})

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        # API call with retry logic
        last_exc: Optional[Exception] = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                )
                # print(completion.choices[0].message.content)
                return completion.choices[0].message.content

            except BadRequestError as e:
                # 400 Bad Request is treated as non-retryable.
                logger.warning(
                    "[%s] 400 Bad Request (not retryable): %s",
                    self.model_name, e.message,
                )
                return None

            except Exception as e:
                last_exc = e
                if self._is_retryable(e) and attempt < MAX_RETRIES:
                    delay = min(
                        RETRY_BASE_DELAY * (2 ** attempt),
                        RETRY_MAX_DELAY,
                    )
                    logger.warning(
                        "[%s] Retryable error (attempt %d/%d), retrying in %.1fs: %s",
                        self.model_name, attempt + 1, MAX_RETRIES, delay, e,
                    )
                    time.sleep(delay)
                else:
                    # Stop on non-retryable failures or after exhausting retries.
                    logger.error(
                        "[%s] API call failed after %d attempts: %s",
                        self.model_name, attempt + 1, e,
                    )
                    return None

        # Safety fallback.
        logger.error("[%s] All retries exhausted: %s", self.model_name, last_exc)
        return None
