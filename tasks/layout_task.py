"""Layout tasks: Balance, Layering, Whitespace, and Alignment."""

from typing import Any, Dict, Optional

from tasks.base_task import BaseTask


class LayoutTask(BaseTask):

    def __init__(self, task_config: Optional[Dict[str, Any]] = None):
        super().__init__(task_config)

        self.balance_prompt = (
            "Balance is the distribution of visual weight in design. "
            "It can be symmetrical (with equal weights on both sides) or "
            "asymmetrical (with unequal weights but still achieving visual balance). "
            "Does element have balance issue?"
        )
        self.layering_prompt = (
            "Use size, color, contrast, and other visual cues to establish "
            "a hierarchical structure of design elements to guide the audience's eyes. "
            "Does element have layering issue?"
        )
        self.whitespace_prompt = (
            "White space refers to the blank area around the elements in a design. "
            "Effective utilization of white space can create balance, visual hierarchy, "
            "and clarity. Does element have whitespace issue?"
        )
        self.alignment_prompt = (
            "Alignment refers to the arrangement of design elements relative to "
            "each other or a specific axis or grid. Proper alignment creates a sense "
            "of order and organization, making the design easier to understand and navigate. "
            "Does element have alignment issue?"
        )
