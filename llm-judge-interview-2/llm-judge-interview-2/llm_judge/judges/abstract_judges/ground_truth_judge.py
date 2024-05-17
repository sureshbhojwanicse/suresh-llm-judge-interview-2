from typing import List
from llm_judge.database.models import Question, Answer, Judgment
from llm_judge.judges.abstract_judges.judge import Judge


class GroundTruthJudge(Judge):
    """
    Judge abstract class for any concrete judges that need to compare against either given or generated ground truth.
    """

    def make_pairs(
        self, questions: List[Question], answers: List[Answer]
    ) -> List[Judgment]:
        """
        Make pairwise comparisons between the baseline LLM and all other LLMs for each question.
        comparison_answers is a list of length 1 containing a baseline_llm answer id.
        """
        llm_answers = self.group_answers_by_llm_id(answers)
        candidate_llms = list(llm_answers.keys())

        judgment_pairs = []
        for question in questions:
            ground_truths = question.ground_truth
            for llm in candidate_llms:
                candidate_answers = llm_answers[llm][question.id]
                for ground_truth, candidate_answer in zip(
                    ground_truths.values(), candidate_answers
                ):
                    # Only append ones where there is a Ground Truth answer
                    if ground_truth:
                        judgment_pairs.append(
                            Judgment(
                                question_id=question.id,
                                answer_id=candidate_answer.id,
                                judge_prompts=self.judge_prompts,
                                judge_type=self.get_type(),
                                ground_truth=ground_truth,
                                comparison_answers=[],
                                score={llm: -float("inf")},
                            )
                        )
        return judgment_pairs

    def set_score(self, pair: Judgment, verdict: str) -> None:
        """
        Set the score of the pair based on the verdict.
        We always set the baseline LLM score to 0.5.
        """
        answer_llm_id = [key for key in pair.score.keys()][0]
        super().set_score_for_llm_id(answer_llm_id, pair, verdict)
