import json
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime


# Function to convert JSON data with MongoDB specific formats
def convert_mongo_json(doc):
    if isinstance(doc, dict):
        for key, value in doc.items():
            if isinstance(value, dict):
                if "$oid" in value:
                    doc[key] = ObjectId(value["$oid"])
                elif "$date" in value:
                    doc[key] = datetime.fromisoformat(
                        value["$date"].replace("Z", "+00:00")
                    )
                else:
                    convert_mongo_json(value)
            elif isinstance(value, list):
                for i in range(len(value)):
                    value[i] = convert_mongo_json(value[i])
    elif isinstance(doc, list):
        for i in range(len(doc)):
            doc[i] = convert_mongo_json(doc[i])
    return doc
