"""
Task registry.

Keys in ``GT.json`` use the format ``"dimension-task_name"``, for example
``"layout-balance"``. The ``dimension`` part maps to the keys in this registry.
"""

from typing import Dict, Type

from tasks.base_task import BaseTask
from tasks.layout_task import LayoutTask
from tasks.typography_task import TypographyTask
from tasks.graphics_task import GraphicsTask
from tasks.color_task import ColorTask

TASK_REGISTRY: Dict[str, Type[BaseTask]] = {
    "layout":  LayoutTask,
    "font":    TypographyTask,
    "graphic": GraphicsTask,
    "color":   ColorTask,
}
