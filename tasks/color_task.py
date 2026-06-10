"""Color tasks: Harmony, Contrast, Appeal, and Psychology."""

from typing import Any, Dict, Optional

from tasks.base_task import BaseTask


class ColorTask(BaseTask):

    def __init__(self, task_config: Optional[Dict[str, Any]] = None):
        super().__init__(task_config)

        self.harmony_prompt = (
            "Color harmony refers to the overall coordination, pleasure and beauty "
            "of the entire color when there are two or more colors in an image. "
            "Does element have harmony issue?"
        )
        self.contrast_prompt = (
            "Color contrast refers to the contrasts, oppositions, and differences "
            "existing among various colors. Does element have contrast issue?"
        )
        self.appeal_prompt = (
            "Color appeal refers to the fact that the selection and combination "
            "of colors can attract the attention of the audience. "
            "Does element have appeal issue?"
        )
        self.psychology_prompt = (
            "Color psychology refers to the idea that color can trigger subjective "
            "psychological experiences and influence emotions, feelings, and behaviors. "
            "Does element have psychology issue?"
        )
