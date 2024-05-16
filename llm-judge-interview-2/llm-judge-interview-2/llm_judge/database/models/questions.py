from typing import Optional, List, Dict, Any
from bson import ObjectId
from pymongo import ReturnDocument
from adapters.types import Conversation
from tqdm import tqdm
import copy

from llm_judge.database.mongo import DatabaseClient
from llm_judge.database.models.base import MongoBaseModel
from llm_judge.utils.common import stringify_dict, stringify_conversation


class Question(MongoBaseModel):
    conversation: Conversation
    conversation_hash: str
    ground_truth: Optional[dict[str, str]] = {}
    args: Optional[dict] = {}

    def serialize(self) -> dict:
        return {
            "id": self.id,
            "conversation": stringify_conversation(self.conversation),
            "conversation_hash": self.conversation_hash,
            "ground_truth": self.ground_truth,
            "args": stringify_dict(self.args),
        }

    def is_classified(self, class_names: List[str]) -> bool:
        """
        Check if the question has been classified by the classifier.
        This function assumes that the classifier stores class names in a list in the args field.
        """
        if "classes" in self.args:
            for class_name in class_names:
                if class_name in self.args["classes"]:
                    return True
        return False

    def is_class(self, class_name: str) -> bool:
        """
        Check if the question has been classified as a specific class by the classifier.
        This function assumes that the classifier stores class names in a list in the args field.
        """
        if "classes" in self.args:
            return class_name in self.args["classes"]

    def save(self) -> "Question":
        """
        Create a new question in the database.
        """
        question_collection = DatabaseClient.get_collection("questions")
        question_collection.insert_one(self.model_dump(by_alias=True))
        return self

    @staticmethod
    def get_question_by_id(question_id: str | ObjectId) -> "Question":
        """
        Retrieve a question by its ID.
        """
        question_collection = DatabaseClient.get_collection("questions")
        question = question_collection.find_one({"_id": ObjectId(question_id)})
        return Question(**question)

    @staticmethod
    def get_question_by_hash(conversation_hash: str) -> "Question":
        """
        Retrieve a question by its conversation hash.
        """
        question_collection = DatabaseClient.get_collection("questions")
        question = question_collection.find_one(
            {"conversation_hash": conversation_hash}
        )
        if question:
            return Question(**question)

    @staticmethod
    def get_questions_by_id(question_ids: list[str | ObjectId]) -> list["Question"]:
        """
        Retrieve a question by its ID.
        """
        question_collection = DatabaseClient.get_collection("questions")
        questions = question_collection.find(
            {"_id": {"$in": [ObjectId(question_id) for question_id in question_ids]}}
        )
        # Ensure its in the same order as the input
        questions = [Question(**question) for question in questions]
        question_id_to_question = {str(question.id): question for question in questions}
        return [
            question_id_to_question[str(question_id)] for question_id in question_ids
        ]

    @staticmethod
    def get_questions_by_hash(conversation_hashs: list[str]) -> list["Question"]:
        """
        Retrieve a question by its conversation hash.
        """
        question_collection = DatabaseClient.get_collection("questions")
        questions = question_collection.find(
            {"conversation_hash": {"$in": conversation_hashs}}
        )
        return [Question(**question) for question in questions if question]

    def update(self) -> "Question":
        """
        Update the question in the database with self, including args and ground truth.
        """
        original_question = self.get_question_by_hash(self.conversation_hash)
        assert (
            original_question
        ), f"question with hash {self.conversation_hash} not found in database"

        updated_question = Question._update_or_none(original_question, self)
        if updated_question:
            question_collection = DatabaseClient.get_collection("questions")
            updated_question = question_collection.find_one_and_update(
                {"_id": updated_question.id},
                {"$set": updated_question.model_dump(by_alias=True)},
                return_document=ReturnDocument.AFTER,
            )
            return Question(**updated_question)
        else:
            return original_question

    @staticmethod
    def batch_update_or_save(questions: List["Question"]) -> List["Question"]:
        """
        Update or save a batch of questions.
        If the question already exists in the database, update it; if the question does not exist, save it to the database.

        Returns a list of updated or saved questions.

        """
        already_existing_questions = Question.get_questions_by_hash(
            [question.conversation_hash for question in questions]
        )
        already_existing_question_hashes = [
            question.conversation_hash for question in already_existing_questions
        ]
        # Ones that are not in this list are new questions that need to be saved
        new_questions = [
            question
            for question in questions
            if question.conversation_hash not in already_existing_question_hashes
        ]
        old_questions = [
            question
            for question in questions
            if question.conversation_hash in already_existing_question_hashes
        ]
        # Update the existing ones
        updated_questions = Question.batch_update(old_questions)
        # Save the new ones
        saved_questions = Question.batch_save(new_questions)

        return updated_questions + saved_questions

    @staticmethod
    def batch_save(questions: List["Question"]) -> List["Question"]:
        """
        Save questions in a batch, instead of individually
        """
        if len(questions) == 0:
            return questions
        question_collection = DatabaseClient.get_collection("questions")
        question_collection.insert_many(
            [question.model_dump(by_alias=True) for question in questions]
        )
        return questions

    @staticmethod
    def _update_dict_if_needed(
        original: Dict[str, Any], updates: Dict[str, Any]
    ) -> bool:
        """
        Update the original dictionary with values from the updates dictionary where they differ.
        Returns True if any updates were made, False otherwise.
        """
        updated = False
        for key, value in updates.items():
            if original.get(key) != value:
                original[key] = value
                updated = True
        return updated


    @staticmethod
    def _update_or_none(
        original_question: "Question", new_question: "Question"
    ) -> Optional["Question"]:
        """
        Update the ground truth and args field of the original question with values from the new question if they are different.
        Returns the original question if it was updated, otherwise returns None.
        """
        updated = Question._update_dict_if_needed(
            original_question.ground_truth, new_question.ground_truth
        )
        updated |= Question._update_dict_if_needed(
            original_question.args, new_question.args
        )

        return original_question if updated else None

    @staticmethod
    def batch_update(questions: List["Question"]) -> List["Question"]:
        """
        Update a batch of questions whose hashes are in the database if there are any differences.
        Return a list of updated questions.
        """
        question_collection = DatabaseClient.get_collection("questions")
        question_hashes = [question.conversation_hash for question in questions]
        original_questions = Question.get_questions_by_hash(question_hashes)
        hash_to_original_question_map = {
            str(question.conversation_hash): question for question in original_questions
        }
        updated_questions = []

        for new_question in questions:
            assert (
                str(new_question.conversation_hash) in hash_to_original_question_map
            ), f"question with hash {new_question.conversation_hash} not found in database"
            original_question = hash_to_original_question_map[
                str(new_question.conversation_hash)
            ]
            updated_question = Question._update_or_none(original_question, new_question)
            if updated_question:
                updated_question = question_collection.find_one_and_update(
                    {"_id": updated_question.id},
                    {"$set": updated_question.model_dump(by_alias=True)},
                    return_document=ReturnDocument.AFTER,
                )
                updated_questions.append(Question(**updated_question))
            else:
                updated_questions.append(original_question)

        return updated_questions
