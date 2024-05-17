"""
This script is setup to dump the martian database transactions to disk. 

This dumps to pickle as ObjectID is not serializable to JSON. 

"""

import os

import pymongo
import pickle
from jsonargparse import ActionConfigFile, ArgumentParser
from dotenv import load_dotenv

load_dotenv("./.env", override=True)


def get_users_by_email(email: str, connection_string: str) -> list:
    client = pymongo.MongoClient(connection_string)
    db = client["backend"]["users"]
    results = []
    results += list(db.find({"email": {"$regex": email}}))
    return results


def get_members_by_user_id(user_ids: list, connection_string: str) -> list:
    client = pymongo.MongoClient(connection_string)
    member_db = client["backend"]["members"]
    members = list(member_db.find({"user": {"$in": user_ids}}))
    return members


def get_orgs_by_member_id(member_ids: list, connection_string: str) -> list:
    client = pymongo.MongoClient(connection_string)
    org_db = client["backend"]["organizations"]
    orgs = list(org_db.find({"members": {"$in": member_ids}}))
    return orgs


def get_transactions_by_org_id(
    org_ids: list, limit: int, connection_string: str
) -> list:
    client = pymongo.MongoClient(connection_string)
    db = client["backend"]["transactions_v2"]
    results = []
    results += list(db.find({"org_local_id": {"$in": org_ids}}).limit(limit))
    return results


def main(
    email_regex: str, connection_string: str, limit: int, output_file: str
) -> None:
    users = get_users_by_email(email_regex, connection_string)
    user_ids = [user["_id"] for user in users]
    members = get_members_by_user_id(user_ids, connection_string)
    member_ids = [member["_id"] for member in members]
    orgs = get_orgs_by_member_id(member_ids, connection_string)
    org_ids = [org["local_id"] for org in orgs]
    transactions = get_transactions_by_org_id(org_ids, limit, connection_string)
    print(f"Found {len(transactions)} transactions")
    with open(output_file, "wb") as f:
        pickle.dump(transactions, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--email_regex", type=str)
    parser.add_argument("--limit", type=int, default=1000000)
    parser.add_argument(
        "--connection_string", type=str, default=os.environ["MARTIAN_MONGO_URI"]
    )
    parser.add_argument("--output_file", type=str, default="transaction_dump.pkl")
    parser.add_argument("--config", action=ActionConfigFile)
    args = parser.parse_args()
    main(args.email_regex, args.connection_string, args.limit, args.output_file)
