import os
from dotenv import load_dotenv, find_dotenv
from pymongo.server_api import ServerApi
from pymongo import MongoClient, InsertOne
from pymongo.database import Database
import json
from pathlib import Path

DATABASE_NAME = "llm-judge"
COLLECTIONS = {"questions", "answers", "judgments", "prompts", "evals"}

load_dotenv(find_dotenv(), override=True)


class DatabaseClient:
    client: MongoClient
    db: Database

    @classmethod
    def connect(cls):
        cls.client = MongoClient(os.environ["MONGO_URI"], server_api=ServerApi("1"))
        cls.db = cls.client[DATABASE_NAME]

    @classmethod
    def disconnect(cls):
        cls.client.close()

    @classmethod
    def get_collection(cls, collection_name: str):
        if collection_name not in COLLECTIONS:
            raise ValueError(f"Collection {collection_name} does not exist")
        return cls.db[collection_name]


DatabaseClient.connect()

# tried to import the given prompts
# # prompt_file_path = Path()
# requesting = []
# with open("../llm-judge.prompts.json", "r") as f:
#     myDict = json.loads(f.read())
#     prompt_collection = DatabaseClient.get_collection("prompts")
#     prompt_collection.insert_many(myDict)
