from typing import Any, Optional
import re

from llm_judge.utils.types import JudgeType
from llm_judge.database.models.judgments import Judgment
from llm_judge.judges.abstract_judges.ground_truth_judge import GroundTruthJudge
from llm_judge.database.models.answers import Answer


class ExactMatchJudge(GroundTruthJudge):
    """
    Judge that assigns scores based on whether the candidate answer exactly matches the reference answer.
    """

    def __init__(
        self,
        judge_prompts: dict[str, str] = {},
        judge_llm: str = "Direct Comparison",
        judge_llm_params: dict[str, Any] = {},
        num_retries: int = 3,
    ):
        super().__init__(judge_prompts, judge_llm, judge_llm_params, num_retries)

    def get_type(self) -> JudgeType:
        return JudgeType.EXACT_MATCH

    @staticmethod
    def parse_answer(answer: str) -> Optional[str]:
        """
        Given the long answer, parse the exact match answer.
        This should be customized based on the dataset.
        """
        match = re.search(r"\[\[\[The final answer is: (.*?)\.", answer)
        if match:
            extracted_answer = match.group(1).strip()
            if extracted_answer in ["yes", "no", "maybe"]:
                return extracted_answer

    def generate_single_judgment(self, pair: Judgment) -> Optional[Judgment]:
        reference_answer = pair.ground_truth
        answer = Answer.get_answer_by_id(pair.answer_id)
        extracted_answer = self.parse_answer(answer.content)
        verdict = "A" if reference_answer == extracted_answer else "B"
        pair.judgments = [
            f"Exact match comparison: reference answer = {reference_answer}, extracted llm answer = {extracted_answer}"
        ]
        self.set_score(pair, verdict)
        pair.save()

        return pair
