from datetime import datetime
from bson import ObjectId
from typing import List, Optional

from llm_judge.database.models.base import MongoBaseModel
from llm_judge.database.mongo import DatabaseClient
from llm_judge.utils.common import stringify_dict


class Prompt(MongoBaseModel):
    description: Optional[str] = None
    tags: List[str] = []
    last_edited: datetime = datetime.now()
    content: str
    args: dict = {}

    def serialize(self) -> dict:
        return {
            "id": self.id,
            "description": self.description if self.description else "",
            "tags": ", ".join(self.tags),
            "last_edited": str(self.last_edited),
            "content": self.content,
            "args": stringify_dict(self.args),
        }

    @staticmethod
    def get_prompt_by_id(prompt_id: str) -> "Prompt":
        """
        Retrieve a prompt by its ID.
        """
        prompt_collection = DatabaseClient.get_collection("prompts")
        prompt = prompt_collection.find_one({"_id": ObjectId(prompt_id)})
        return Prompt(**prompt)

    def get_or_save(self) -> "Prompt":
        """
        Create a new prompt in the database.
        """
        prompt_collection = DatabaseClient.get_collection("prompts")
        model_data = self.model_dump(by_alias=True)
        model_data = {
            k: v for k, v in model_data.items() if k not in ["_id", "id", "last_edited"]
        }
        existing_document = prompt_collection.find_one(model_data)
        if existing_document:
            self.id = existing_document["_id"]
        else:
            insert_result = prompt_collection.insert_one(self.model_dump(by_alias=True))
            self.id = insert_result.inserted_id
        return self
