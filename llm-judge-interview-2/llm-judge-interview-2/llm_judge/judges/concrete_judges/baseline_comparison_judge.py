from typing import List, Optional, Any
from bson import ObjectId

from llm_judge.judges.abstract_judges.judge import Judge
from llm_judge.database.models.questions import Question
from llm_judge.database.models.answers import Answer
from llm_judge.database.models.judgments import Judgment
from llm_judge.utils.types import JudgeType


class BaselineComparisonJudge(Judge):
    """
    Judge that compares a candidate LLM to a baseline LLM to determine which is better.
    """

    def __init__(
        self,
        baseline_llm: str,
        judge_prompts: dict[str, str],
        judge_llm: str = "gpt-4-turbo-preview",
        judge_llm_params: dict[str, Any] = {"temperature": 0},
        num_retries: int = 3,
    ):
        self.baseline_llm = baseline_llm
        self.judge_llm_params = judge_llm_params
        for k, v in judge_prompts.items():
            judge_prompts[k] = ObjectId(v)
        self.judge_prompts = judge_prompts
        self.judge_llm = judge_llm
        self.num_retries = num_retries

    def get_type(self) -> JudgeType:
        return JudgeType.BASELINE_PAIRWISE

    def make_pairs(
        self, questions: List[Question], answers: List[Answer]
    ) -> List[Judgment]:
        """
        Make pairwise comparisons between the baseline LLM and all other LLMs for each question.
        comparison_answers is a list of length 1 containing a baseline_llm answer id.
        """
        llm_answers = self.group_answers_by_llm_id(answers)
        all_llms = list(llm_answers.keys())
        candidate_llms = [x for x in all_llms if x != self.baseline_llm]

        judgment_pairs = []
        for question in questions:
            baseline_answers = self.sort_answers_by_id(
                llm_answers[self.baseline_llm][question.id]
            )
            for llm in candidate_llms:
                candidate_answers = self.sort_answers_by_id(
                    llm_answers[llm][question.id]
                )
                for baseline_answer, candidate_answer in zip(
                    baseline_answers, candidate_answers
                ):
                    judgment_pairs.append(
                        Judgment(
                            question_id=question.id,
                            answer_id=candidate_answer.id,
                            judge_prompts=self.judge_prompts,
                            judge_type=self.get_type(),
                            comparison_answers=[baseline_answer.id],
                            score={
                                self.baseline_llm: -float("inf"),
                                llm: -float("inf"),
                            },
                        )
                    )
        return judgment_pairs

    def set_score(self, pair: Judgment, verdict: str) -> None:
        """
        Set the score of the pair based on the verdict.
        We always set the baseline LLM score to 0.5.
        """
        pair.score[self.baseline_llm] = 0.5
        answer_llm_id = [key for key in pair.score.keys() if key != self.baseline_llm][
            0
        ]
        super().set_score_for_llm_id(answer_llm_id, pair, verdict)

    def generate_single_judgment(self, pair: Judgment) -> Optional[Judgment]:
        """
        Generate a single judgment for a pair of answers.
        """
        reference_answer = Answer.get_answer_by_id(pair.comparison_answers[0]).content
        return super()._generate_single_judgment(pair, reference_answer)
