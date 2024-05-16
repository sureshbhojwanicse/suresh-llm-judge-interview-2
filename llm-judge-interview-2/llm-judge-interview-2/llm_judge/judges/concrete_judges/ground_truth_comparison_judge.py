from typing import Optional, Any
from bson import ObjectId

from llm_judge.database.models.judgments import Judgment
from llm_judge.utils.types import JudgeType
from llm_judge.judges.abstract_judges.ground_truth_judge import GroundTruthJudge


class GroundTruthComparisonJudge(GroundTruthJudge):
    """
    Judge that assigns scores by using an LLM to decide whether the candidate answer is better than the ground truth.
    """

    def __init__(
        self,
        judge_prompts: dict[str, str],
        judge_llm: str = "gpt-4-turbo-preview",
        judge_llm_params: dict[str, Any] = {"temperature": 0},
        num_retries: int = 3,
    ):
        self.judge_llm_params = judge_llm_params
        for k, v in judge_prompts.items():
            judge_prompts[k] = ObjectId(v)
        self.judge_prompts = judge_prompts
        self.judge_llm = judge_llm
        self.num_retries = num_retries

    def get_type(self) -> JudgeType:
        return JudgeType.GENERATIVE_GROUND_TRUTH

    def generate_single_judgment(self, pair: Judgment) -> Optional[Judgment]:
        reference_answer = pair.ground_truth
        return super()._generate_single_judgment(pair, reference_answer)
