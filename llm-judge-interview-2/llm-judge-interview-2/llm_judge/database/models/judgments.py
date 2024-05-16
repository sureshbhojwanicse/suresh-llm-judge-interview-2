from typing import Optional, List
from bson import ObjectId

from llm_judge.database.mongo import DatabaseClient
from llm_judge.database.models.base import MongoBaseModel
from llm_judge.utils.common import stringify_dict


class Judgment(MongoBaseModel):
    question_id: ObjectId
    answer_id: ObjectId
    judge_prompts: dict[str, ObjectId]
    judge_type: str
    comparison_answers: Optional[List[ObjectId]] = None
    ground_truth: Optional[str] = None
    judgments: Optional[List[str]] = None
    score: Optional[dict[str, float]] = None
    args: Optional[dict] = {}

    def serialize(self) -> dict:
        return {
            "id": self.id,
            "question_id": self.question_id,
            "answer_id": self.answer_id,
            "judge_prompts": self.judge_prompts,
            "judge_type": self.judge_type,
            "comparison_answers": self.comparison_answers,
            "ground_truth": self.ground_truth,
            "judgments": self.judgments,
            "score": stringify_dict(self.score),
            "args": stringify_dict(self.args),
        }

    def get_judgment_by_args(self) -> Optional[List["Judgment"]]:
        judgment_collection = DatabaseClient.get_collection("judgments")
        query = {
            "question_id": self.question_id,
            "answer_id": self.answer_id,
            "judge_prompts": self.judge_prompts,
            "judge_type": self.judge_type,
            "comparison_answers": {
                "$all": self.comparison_answers,
                "$size": len(self.comparison_answers),
            },
            "ground_truth": self.ground_truth,
        }

        judgments = judgment_collection.find(query)
        if judgments:
            return [Judgment(**judgment) for judgment in judgments]

    def save(self) -> "Judgment":
        judgment_collection = DatabaseClient.get_collection("judgments")
        judgment_collection.insert_one(self.model_dump(by_alias=True))
        return self

    # added this method to extract single judgement object for the prompt_tuner.py's training data preparation process
    @staticmethod
    def get_judgment_by_id(judgment_id: str) -> "Judgment":
        judgment_collection = DatabaseClient.get_collection("judgments")
        judgment = judgment_collection.find_one({"_id": ObjectId(judgment_id)})
        return Judgment(**judgment)

    @staticmethod
    def get_judgments_by_id(judgment_ids: list[str]) -> list["Judgment"]:
        judgment_collection = DatabaseClient.get_collection("judgments")
        judgments = judgment_collection.find(
            {"_id": {"$in": [ObjectId(answer_id) for answer_id in judgment_ids]}}
        )
        judgments = [Judgment(**judgment) for judgment in judgments]
        judgment_id_to_question = {str(judgment.id): judgment for judgment in judgments}
        return [
            judgment_id_to_question[str(judgment_id)] for judgment_id in judgment_ids
        ]
