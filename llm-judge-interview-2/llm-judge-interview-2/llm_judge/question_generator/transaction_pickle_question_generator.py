import random
from typing import Any, Optional, List
from adapters.types import Conversation
import pickle

from llm_judge.question_generator.base_question_generator import BaseQuestionGenerator
from llm_judge.database.models.questions import Question
from llm_judge.utils.common import stringify_conversation, gen_hash


class TransactionPickleQuestionGenerator(BaseQuestionGenerator):
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
        with open(self.data_path, "rb") as f:
            data = pickle.load(f)
        return data

    def get_conversation_from_entry(self, entry: Any) -> Conversation:
        """
        Extract a conversation from a data entry.
        """
        return Conversation(entry["user_request"]["input"])

    def get_args_from_entry(self, entry: Any) -> dict:
        """
        Extract additional arguments from a data entry.
        """
        args = {}
        for key in entry:
            if key != "conversation":
                args[key] = entry[key]
        return args

    def make_questions(self) -> List[Question]:
        """
        Generate a list of question objects from the data.
        """
        data = self.load_data()
        # Filter empty conversations
        data = [
            entry
            for entry in data
            if "user_request" in entry
            and "input" in entry["user_request"]
            and "turns" in entry["user_request"]["input"]
            and len(entry["user_request"]["input"]["turns"]) > 0
            and entry["user_request"]["input"]["turns"][0]["content"] != "None"
        ]

        random.seed(self.random_seed)
        random.shuffle(data)
        if self.num_samples and self.num_samples < len(data):
            data = data[: self.num_samples]
        questions = []
        for entry in data:
            conversation = self.get_conversation_from_entry(entry)
            questions.append(
                Question(
                    conversation=conversation,
                    conversation_hash=gen_hash(stringify_conversation(conversation)),
                    args=self.get_args_from_entry(entry),
                )
            )
        return questions
