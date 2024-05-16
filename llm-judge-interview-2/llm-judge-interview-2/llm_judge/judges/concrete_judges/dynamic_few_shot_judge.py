from typing import List, Optional, Any
from bson import ObjectId
from typing import List, Optional, Any
from bson import ObjectId
import traceback
from collections import defaultdict
from adapters.types import Conversation, Turn, ConversationRole
from adapters.adapter_factory import AdapterFactory
from adapters import AdapterRateLimitException

from llm_judge.judges.abstract_judges.judge import Judge
from llm_judge.database.models.prompts import Prompt
from llm_judge.database.models.questions import Question
from llm_judge.database.models.answers import Answer
from llm_judge.database.models.judgments import Judgment
from llm_judge.utils.types import JudgeType

import pandas as pd
import voyageai
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import dotenv

dotenv.load_dotenv("../../../.env")
global_voyage_client = voyageai.Client()


class DynamicFewShotJudge(Judge):
    """
    Judge that compares a candidate LLM to a baseline LLM to determine which is better.
    """

    def __init__(
        self,
        baseline_llm: str,
        ideal_judgment_path: str,
        judge_prompts: dict[str, str],
        judge_llm: str = "gpt-4-turbo-preview",
        judge_llm_params: dict[str, Any] = {"temperature": 0},
        num_retries: int = 3,
        embedding_model: str = "voyage-large-2-instruct",
    ):
        self.baseline_llm = baseline_llm
        self.judge_llm_params = judge_llm_params
        for k, v in judge_prompts.items():
            judge_prompts[k] = ObjectId(v)
        self.judge_prompts = judge_prompts
        self.judge_llm = judge_llm
        self.num_retries = num_retries
        self.ideal_judgment_df = pd.read_csv(ideal_judgment_path)
        assert "question" in self.ideal_judgment_df.columns
        assert "ideal_judgment_rationale" in self.ideal_judgment_df.columns

        self.embedding_model = embedding_model
        self.ideal_judgment_df["question_embedding"] = self.embed_strings(
            self.ideal_judgment_df["question"].to_list()
        )

    def get_type(self) -> JudgeType:
        return JudgeType.DYNAMIC_FEW_SHOT

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
        try:
            conversation = self._create_judge_input(pair, "prompt_template")
            winner, judgment_content = self._judge_pair(conversation, self.judge_llm)

            conversation_swapped = self._create_judge_input(
                pair, "reversed_prompt_template"
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
                f"Error generating single judgment in for question id: {pair.question_id}, answer id: {pair.answer_id}, comparison answers: {pair.comparison_answers}  error: {e}"
            )
            print(traceback.format_exc())
            return None

    def _create_judge_input(
        self, pair: Judgment, prompt_template_name: str
    ) -> Conversation:
        """
        Create a conversation using the given prompts and the pair of answers to call the LLM Judge with.
        """
        system_prompt = self._create_system_prompt(pair)
        judge_prompt = Prompt.get_prompt_by_id(
            self.judge_prompts[prompt_template_name]
        ).content
        question = Question.get_question_by_id(pair.question_id)
        baseline_answer = Answer.get_answer_by_id(pair.comparison_answers[0])
        answer = Answer.get_answer_by_id(pair.answer_id)
        user_prompt = judge_prompt.format(
            question=question.conversation.convert_to_prompt(),
            reference=baseline_answer.content,
            candidate=answer.content,
        )
        conversation = Conversation(
            [
                Turn(role=ConversationRole.system, content=system_prompt),
                Turn(role=ConversationRole.user, content=user_prompt),
            ]
        )
        return conversation

    def embed_strings(self, strings: List[str]) -> List[List[float]]:
        if len(strings) > 128:
            print(f"Embedding {len(strings)} strings")
            t = time.time()

        total_output_embeddings = []
        for chunk in self.chunk_list(strings):
            output_embeddings = global_voyage_client.embed(
                chunk, model=self.embedding_model, input_type="document"
            ).embeddings
            total_output_embeddings.extend(output_embeddings)

        if len(strings) > 128:
            print(
                f"Finished embedding {len(strings)} strings, takes {time.time() - t} seconds"
            )
        return total_output_embeddings

    def chunk_list(self, lst, chunk_size=128):
        """Breaks a list into sublists of specified size."""
        return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

    def get_existing_judgments(self, judgment: Judgment) -> Optional[List[Judgment]]:
        return judgment.get_judgment_by_args()

    def _create_system_prompt(self, pair: Judgment) -> str:
        base_prompt_obj = Prompt.get_prompt_by_id(self.judge_prompts["system"])
        # assert "dynamic_few_shot" in base_prompt_obj.tags
        base_prompt = base_prompt_obj.content
        question_content = Question.get_question_by_id(
            pair.question_id
        ).conversation.convert_to_readable_string()
        few_shot_prompt = self._create_few_shot_example(question_content)
        return base_prompt + few_shot_prompt

    def _create_few_shot_example(self, question: str, n: int = 1) -> str:
        question_embedding = self.embed_strings([question])[0]
        similarities = cosine_similarity(
            np.array([question_embedding]),
            np.stack(self.ideal_judgment_df["question_embedding"]),
        )[0]
        top_n_indices = np.argsort(similarities)[-n:][
            ::-1
        ]  # Get top n indices in descending order
        top_n_judgments = self.ideal_judgment_df.iloc[top_n_indices]

        few_shot_prompt = "\n The following are some similar prompts that should serve as baseline examples for you to judge the candidate answer. \n"
        for row in top_n_judgments.iterrows():
            row_dict = row[1].to_dict()
            few_shot_prompt += f"""
--- judgment example ---
[Conversation]
{row_dict['question']}

[The Start of Reference Answer]
{row_dict['ground_truth']}
[The End of Reference Answer]

[The Start of Candidate Answer]
{row_dict['answer']}
[The End of Candidate Answer]

[Reference Judgment] 
{row_dict["ideal_judgment_rationale"]}
[The End of Judgment]
---
            """
        return few_shot_prompt


if __name__ == "__main__":
    import pickle, os

    os.chdir("/Users/jasonmartian/Desktop/llm-judge")
    import dotenv

    dotenv.load_dotenv()
    judge = DynamicFewShotJudge(
        baseline_llm="gpt-3.5-turbo-0125",
        ideal_judgment_path="data/DeepAI/test-2024-04-30-17-52/enriched_judgments-ideal-v2.csv",
        judge_prompts={
            "system": "662813e0e25b6076a9e03df8",
            "prompt_template": "662821e23eb9ef01018e30e2",
            "reversed_prompt_template": "6628224eb84c0693351ca6a4",
        },
    )

    # Example of testing if an instance is pickleable
    try:
        # Assuming `judge` is an instance of DynamicFewShotJudge
        pickle.dumps(judge)
        print("Judge is pickleable!")
    except pickle.PicklingError as e:
        print("Pickle error:", e)
