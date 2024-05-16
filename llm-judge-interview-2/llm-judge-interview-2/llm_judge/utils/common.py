from typing import List
import csv
import hashlib
import json
import yaml
import os
from adapters.types import Conversation

from llm_judge.database.models.base import MongoBaseModel
from llm_judge.utils.types import LLMParamsList


def stringify_conversation(conversation: Conversation) -> str:
    """
    Convert a conversation object to a readable string.
    """
    return "".join([f"{turn.role}: {turn.content}\n" for turn in conversation.turns])


def stringify_dict(args: dict) -> str:
    """
    Convert a dictionary to a readable string.
    """
    return "".join([f"{key}: {value}\n" for key, value in args.items()])


def stringify_llm_params(params: LLMParamsList) -> str:
    """
    Convert a list of dictionaries to a readable string.
    """
    serialized_data = []
    for param_dict in params:
        serialized_dict = {key: value.model_dump() for key, value in param_dict.items()}
        serialized_data.append(serialized_dict)

    return json.dumps(serialized_data, sort_keys=True)


def gen_hash(content: str) -> str:
    """
    Generate a hash from a string.
    """
    hash_object = hashlib.sha256()
    hash_object.update(content.encode("utf-8"))
    return hash_object.hexdigest()


def write_to_csv(objects: List[MongoBaseModel], fp: str) -> None:
    """
    Write a list of MongoBaseModel objects to a CSV file in readable form.
    """
    if not objects:
        return
    if os.path.exists(fp):
        print(f"Note: The file {fp} already exists and will be replaced.")
    with open(fp, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=objects[0].get_seralization_header())
        writer.writeheader()
        for object in objects:
            writer.writerow(object.serialize())
    print(f"Wrote {len(objects)} objects to {fp}")



def write_df_to_csv(df, fp: str, write_index=False) -> None:
    from pandas import DataFrame

    DataFrame(df).to_csv(fp, index=write_index)
    print(f"Wrote DataFrame with {len(df)} rows to {fp}")


def write_ids_to_json(objects: List[MongoBaseModel], fp: str) -> None:
    """
    Write a list of MongoBaseModel objects' IDs to a JSON file.
    """
    ids = [str(object.id) for object in objects]
    with open(fp, "w") as f:
        json.dump(ids, f, ensure_ascii=False)


def load_from_json(path: str) -> List[str]:
    """
    Load a list of strings from a JSON file.
    """
    with open(path, "r") as f:
        return json.load(f)


def combine_and_deduplicate(
    list1: List[MongoBaseModel], list2: List[MongoBaseModel]
) -> List[MongoBaseModel]:
    """
    Combine two lists of MongoBaseModel objects and deduplicate them.
    """
    combined = list1 + list2
    return list({object.id: object for object in combined}.values())


def save_experiment_config(config_path: str, content: str) -> None:
    """
    Save the experiment configuration to a file."""
    with open(config_path, "w") as writer:
        writer.write(content)


def parse_yaml_file(path) -> None:
    with open(path, "r") as file:
        return yaml.safe_load(file)
