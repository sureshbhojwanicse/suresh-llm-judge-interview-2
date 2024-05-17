from abc import ABC, abstractmethod
from typing import List, Any, Optional
import traceback
from bson import ObjectId
import time
from collections import defaultdict
from adapters.types import Conversation, Turn, ConversationRole
from adapters.adapter_factory import AdapterFactory
from adapters import AdapterRateLimitException

from llm_judge.utils.types import JudgeType
from llm_judge.database.models.questions import Question
from llm_judge.database.models.answers import Answer
from llm_judge.database.models.judgments import Judgment
from llm_judge.database.models.prompts import Prompt


class Judge(ABC):
    def __init__(
        self,
        judge_prompts: dict[str, str],
        judge_llm: str = "gpt-4-turbo-preview",
        judge_llm_params: dict[str, Any] = {"temperature": 0},
        num_retries: int = 3,
    ):
        for k, v in judge_prompts.items():
            judge_prompts[k] = ObjectId(v)
        self.judge_prompts = judge_prompts
        self.judge_llm = judge_llm
        self.judge_llm_params = judge_llm_params
        self.num_retries = num_retries

    @abstractmethod
    def get_type(self) -> JudgeType:
        """
        Return the type of the judge.
        """
        raise NotImplementedError("get_type not implemented")

    @abstractmethod
    def generate_single_judgment(self):
        """
        Generate a single judgment given self.
        """
        raise NotImplementedError("generate_single_judgment not implemented")

    @abstractmethod
    def make_pairs(self, questions: List[Question], answers: List[Answer], **kwargs):
        """
        Create Judgment objects for a list of questions and answers using the given judge.
        """
        raise NotImplementedError("make_pairs not implemented")

    @abstractmethod
    def set_score(self, pair: Judgment, verdict: str) -> None:
        """
        Set the score of the pair based on the verdict.
        """
        raise NotImplementedError("set_score not implemented")

    @staticmethod
    def group_answers_by_llm_id(
        answers: List[Answer],
    ) -> dict[str, dict[ObjectId, List[Answer]]]:
        """
        Group answers by llm_id and question_id.
        We account for the prompt modification of an answer in the llm_id.
        """
        llm_answers = defaultdict(lambda: defaultdict(list))
        for answer in answers:
            modification = "-".join(list(answer.prompt_modification.keys()))
            llm = f"{answer.llm_id}-{modification}" if modification else answer.llm_id
            llm_answers[llm][answer.question_id].append(answer)
        return llm_answers

    @staticmethod
    def sort_answers_by_id(answers: List[Answer]) -> List[Answer]:
        return sorted(answers, key=lambda x: x.id)

    def _create_judge_input(
        self, pair: Judgment, prompt_template_name: str, reference_answer: str
    ) -> Conversation:
        """
        Create a conversation using the given prompts and the pair of answers to call the LLM Judge with.
        """
        system_prompt = Prompt.get_prompt_by_id(self.judge_prompts["system"]).content
        judge_prompt = Prompt.get_prompt_by_id(
            self.judge_prompts[prompt_template_name]
        ).content
        question = Question.get_question_by_id(pair.question_id)
        answer = Answer.get_answer_by_id(pair.answer_id)
        user_prompt = judge_prompt.format(
            question=question.conversation.convert_to_prompt(),
            reference=reference_answer,
            candidate=answer.content,
        )
        conversation = Conversation(
            [
                Turn(role=ConversationRole.system, content=system_prompt),
                Turn(role=ConversationRole.user, content=user_prompt),
            ]
        )
        return conversation

    @staticmethod
    def _parse_winner_from_judgment(judgment: str) -> str:
        if "[[A]]" in judgment:
            return "A"
        elif "[[B]]" in judgment:
            return "B"
        elif "[[C]]" in judgment:
            return "C"
        else:
            return "error"

    def _judge_pair(self, conv: Conversation, judge_llm: str) -> tuple[str, str]:
        """
        Judge a single pair of answers using the given LLM by calling adapters.
        """

        # change function call
        llm = AdapterFactory.get_adapter_by_path(judge_llm)
        input_conv = llm.convert_to_input(conv)
        winner = "error"
        judgment = ""

        num_retries = 0
        while winner == "error" and self.num_retries > num_retries:
            try:
                judgment = llm.execute_sync(
                    input_conv, **self.judge_llm_params
                ).response.content
            except AdapterRateLimitException as e:
                print(f"Rate limit exceeded, waiting 1 minute, {e}")
                time.sleep(60)

            winner = self._parse_winner_from_judgment(judgment)
            if winner == "error":
                print(f"** regenerate due to error. Judgement: {judgment}")
            num_retries += 1

        return winner, judgment.strip()

    def set_score_for_llm_id(self, llm_id: str, pair: Judgment, verdict: str) -> None:
        """
        Set the score of the given llm id based on the verdict.
        """
        if verdict == "A":
            pair.score[llm_id] = 1.0
        elif verdict == "B":
            pair.score[llm_id] = 0
        elif verdict == "C" or verdict == "unsure":
            pair.score[llm_id] = 0.5

    def _generate_single_judgment(
        self, pair: Judgment, reference_answer: str
    ) -> Optional[Judgment]:
        try:
            conversation = self._create_judge_input(
                pair, "prompt_template", reference_answer
            )
            winner, judgment_content = self._judge_pair(conversation, self.judge_llm)

            conversation_swapped = self._create_judge_input(
                pair, "reversed_prompt_template", reference_answer
            )
            winner_swapped, judgment_content_swapped = self._judge_pair(
                conversation_swapped, self.judge_llm
            )

            if winner == winner_swapped:
                verdict = winner
            elif winner == "error" and winner_swapped != "error":
                verdict = winner_swapped
            elif winner_swapped == "error" and winner != "error":
                verdict = winner
            else:
                verdict = "unsure"

            pair.judgments = [judgment_content, judgment_content_swapped]
            self.set_score(pair, verdict)
            pair.save()
            return pair
        except Exception as e:
            print(
                f"Error generating single judgment in for question id: {pair.question_id}, answer id: {pair.answer_id}, ground truth: {pair.ground_truth}  error: {e}"
            )
            print(traceback.format_exc())
            return None
