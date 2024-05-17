from abc import ABC, abstractmethod
from typing import List
from llm_judge.database.models.questions import Question


class BaseQuestionGenerator(ABC):
    @abstractmethod
    def make_questions(self) -> List[Question]:
        raise NotImplementedError("make_questions not implemented")
