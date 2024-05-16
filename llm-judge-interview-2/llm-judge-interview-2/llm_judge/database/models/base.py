from abc import abstractmethod
from typing import Annotated, List
from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field, PlainSerializer

PyObjectId = Annotated[
    ObjectId,
    PlainSerializer(
        lambda s: str(s),  # pylint: disable=unnecessary-lambda
        return_type=str,
        when_used="json",
    ),
]


class MongoBaseModel(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")

    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    @abstractmethod
    def serialize(self) -> dict:
        pass

    def get_seralization_header(self) -> List[str]:
        return list(self.serialize().keys())
