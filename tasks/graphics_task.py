"""Graphics tasks: Quality and Relevance."""

from typing import Any, Dict, Optional

from tasks.base_task import BaseTask


class GraphicsTask(BaseTask):

    def __init__(self, task_config: Optional[Dict[str, Any]] = None):
        super().__init__(task_config)

        self.quality_prompt = (
            "Resolution is a measure of image detail. "
            "Appropriate resolution ensures the best picture quality and readability. "
            "Does element have quality issue?"
        )
        self.relevance_prompt = (
            "Relevance refers to the direct connection between a graphic element "
            "and the meaning it conveys. Does element have relevance issue?"
        )
