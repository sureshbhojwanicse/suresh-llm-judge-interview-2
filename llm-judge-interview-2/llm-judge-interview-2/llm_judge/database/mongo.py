import os
from dotenv import load_dotenv, find_dotenv
from pymongo.server_api import ServerApi
from pymongo import MongoClient
from pymongo.database import Database
from ..utils.mongoimport import convert_mongo_json
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

# Inserting the given prompt json into db for future querying.
collection = DatabaseClient.get_collection("prompts")
if not collection.count_documents({}) > 0:
    os_path = os.path.abspath("../llm-judge.prompts.json")
    with open(os_path, "r") as file:
        data = json.load(file)
    # Convert the JSON data to MongoDB compatible format
    converted_data = [convert_mongo_json(record) for record in data]
    # insert
    collection.insert_many(converted_data)
