from typing import Optional, List, Any
from bson import ObjectId

from llm_judge.database.mongo import DatabaseClient
from llm_judge.database.models.base import MongoBaseModel
from llm_judge.utils.common import stringify_dict


class Answer(MongoBaseModel):
    question_id: ObjectId
    llm_id: str
    params: dict[str, Any] = {}
    prompt_modification: dict[str, str] = {}
    content: str
    cost: float
    args: Optional[dict] = {}

    def serialize(self) -> dict:
        return {
            "id": self.id,
            "question_id": self.question_id,
            "llm_id": self.llm_id,
            "params": stringify_dict(self.params),
            "prompt_modification": stringify_dict(self.prompt_modification),
            "content": self.content,
            "cost": self.cost,
            "args": stringify_dict(self.args),
        }

    def save(self) -> "Answer":
        answer_collection = DatabaseClient.get_collection("answers")
        answer_collection.insert_one(self.model_dump(by_alias=True))
        return self

    @staticmethod
    def get_answer_by_id(answer_id: str | ObjectId) -> "Answer":
        answer_collection = DatabaseClient.get_collection("answers")
        answer = answer_collection.find_one({"_id": ObjectId(answer_id)})
        return Answer(**answer)

    @staticmethod
    def get_answers_by_id(answer_ids: list[str | ObjectId]) -> list["Answer"]:
        answer_collection = DatabaseClient.get_collection("answers")
        answers = answer_collection.find(
            {"_id": {"$in": [ObjectId(answer_id) for answer_id in answer_ids]}}
        )
        answers = [Answer(**answer) for answer in answers]
        answers_id_to_question = {str(answer.id): answer for answer in answers}
        return [answers_id_to_question[str(answer_id)] for answer_id in answer_ids]

    def get_answers_by_args(self) -> Optional[List["Answer"]]:
        answer_collection = DatabaseClient.get_collection("answers")
        params_conditions = [
            {"params." + key: value} for key, value in self.params.items()
        ]
        prompt_modification_conditions = (
            [
                {"prompt_modification." + key: value}
                for key, value in self.prompt_modification.items()
            ]
            if self.prompt_modification
            else [{"prompt_modification": {}}]
        )

        query = {
            "$and": [
                {"question_id": self.question_id},
                {"llm_id": self.llm_id},
                *params_conditions,
                *prompt_modification_conditions,
            ]
        }
        answers = answer_collection.find(query)
        if answers:
            return [Answer(**answer) for answer in answers]
