from datetime import datetime
from bson import ObjectId
from typing import Optional, List

from llm_judge.database.models.base import MongoBaseModel
from llm_judge.database.mongo import DatabaseClient
from llm_judge.utils.common import stringify_dict


class EvalResult(MongoBaseModel):
    judge_prompt: str
    judge_score: float
    timestamp: datetime = datetime.now()
    args: dict = {}

    # just a serialization of mongodb model
    def serialize(self) -> dict:
        return {
            "id": self.id,
            "judge_prompt": self.judge_prompt,
            "judge_score": self.judge_score,
            "timestamp": str(self.timestamp),
            "args": stringify_dict(self.args),
        }

    # save the initiated class object in the db
    def save(self) -> "EvalResult":
        eval_collection = DatabaseClient.get_collection("evals")
        eval_collection.insert_one(self.model_dump(by_alias=True))
        return self

    # for getting the evaluation class object with the it's id
    @classmethod
    def get_evaluation_result_by_id(eval_id: str) -> "EvalResult":
        """
        Retrieve evaluation results from MongoDB.
        """
        eval_collection = DatabaseClient.get_collection("evals")
        eval = eval_collection.find_one({"_id": ObjectId(eval_id)})
        return EvalResult(**eval)
