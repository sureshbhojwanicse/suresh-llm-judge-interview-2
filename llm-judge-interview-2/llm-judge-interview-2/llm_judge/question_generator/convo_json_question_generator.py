import json
import random
from typing import Any, Optional, List
from adapters.types import Conversation

from llm_judge.question_generator.base_question_generator import BaseQuestionGenerator
from llm_judge.database.models.questions import Question
from llm_judge.utils.common import stringify_conversation, gen_hash


class ConvoJSONQuestionGenerator(BaseQuestionGenerator):
    def __init__(
        self, data_path: str, num_samples: Optional[int] = None, random_seed: int = 42
    ):
        self.data_path = data_path
        self.num_samples = num_samples
        self.random_seed = random_seed

    def load_data(self) -> list[Any]:
        """
        Load the data from the input file.
        """
        with open(self.data_path, "r") as f:
            data = json.load(f)
        return data

    def get_conversation_from_entry(self, entry: Any) -> Conversation:
        """
        Extract a conversation from a data entry.
        """
        return Conversation(entry["conversation"])

    def get_ground_truth_from_entry_or_none(self, entry: Any) -> Optional[str]:
        """
        Extract a ground_truth string from a data entry if it exists
        """
        if "ground_truth" in entry:
            return entry["ground_truth"]

    def get_args_from_entry(self, entry: Any) -> dict:
        """
        Extract additional arguments from a data entry.
        """
        args = {}
        for key in entry:
            if key not in ["conversation", "ground_truth"]:
                args[key] = entry[key]
        return args

    def make_questions(self) -> List[Question]:
        """
        Generate a list of question objects from the data.
        """
        data = self.load_data()

        random.seed(self.random_seed)
        random.shuffle(data)
        if self.num_samples and self.num_samples < len(data):
            data = data[: self.num_samples]
        questions = []

        for entry in data:
            conversation = self.get_conversation_from_entry(entry)
            ground_truth = self.get_ground_truth_from_entry_or_none(entry)
            questions.append(
                Question(
                    conversation=conversation,
                    conversation_hash=gen_hash(stringify_conversation(conversation)),
                    # to be compatible with given data `raw_data.json`
                    ground_truth=(
                        {list(ground_truth.keys())[0]: list(ground_truth.values())[0]}
                        if ground_truth
                        else {}
                    ),
                    args=self.get_args_from_entry(entry),
                )
            )
        return questions
