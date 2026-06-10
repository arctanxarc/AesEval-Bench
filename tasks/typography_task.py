"""Typography tasks: Legibility and Hierarchy."""

from typing import Any, Dict, Optional

from tasks.base_task import BaseTask


class TypographyTask(BaseTask):

    def __init__(self, task_config: Optional[Dict[str, Any]] = None):
        super().__init__(task_config)

        self.legibility_prompt = (
            "Legibility refers to the recognition of individual characters "
            "and the relationships between them when they are arranged side by side. "
            "Does element have legibility issue?"
        )
        self.hierarchy_prompt = (
            "The presentation of the font has a hierarchical structure, "
            "so users can scan the text to obtain key information. "
            "Does element have hierarchy issue?"
        )
