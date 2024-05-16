from typing import Optional, List, Tuple
from bson import ObjectId
import copy
import os
import json
import random
import tiktoken
import traceback
from jsonargparse import ActionConfigFile, ArgumentParser
import concurrent.futures
from tqdm import tqdm
from adapters import AdapterFactory
from adapters.types import Conversation

from llm_judge.database.models.answers import Answer
from llm_judge.database.models.questions import Question
from llm_judge.utils.prompt_modification import make_prompt_modification
from llm_judge.utils.common import (
    write_to_csv,
    write_ids_to_json,
    load_from_json,
    save_experiment_config,
)
from llm_judge.utils.types import LLMParamsList


ENCODING = tiktoken.get_encoding(
    "cl100k_base"
)  # for now, we will use the cl100k_base encoding for all models as an easy estimation
TOKEN_BUFFER = 500  # what we add on top of token count from the encoding as a buffer
NUM_RETRIES = 2  # number of times we retry generating an answer if it fails


def create_answer_objects(question_ids: List[str], llm_params: LLMParamsList):
    """
    Create a list of answer objects given the set of questions, llms, and parameters to generate.
    We use these Answers either to find existing Answers in the database or to generate new ones and fill in the content & cost.
    """
    answers = []
    for question_id in question_ids:
        for config in llm_params:
            for llm_id, params in config.items():
                answer = Answer(
                    question_id=ObjectId(question_id),
                    llm_id=llm_id,
                    params={
                        k: v
                        for k, v in vars(params).items()
                        if k != "prompt_modification"
                    },
                    prompt_modification=(
                        params["prompt_modification"]
                        if "prompt_modification" in params
                        else {}
                    ),
                    content="",
                    cost=0.0,
                )
                answers.append(answer)

    return answers


def get_existing_answers(
    answer_object: Answer, N: Optional[int] = 1
) -> List[Answer] | None:
    """
    Return existing answers that matches the question_id, llm, and params required if they exist.
    """
    answers = answer_object.get_answers_by_args()
    if not answers:
        return None
    random.shuffle(answers)
    return answers[:N]


def shorten_conversation(
    conversation: Conversation, context_length: int
) -> Conversation:
    """
    Shorten the conversation, if necessary, to fit within the context length of the llm.
    """
    prompt_len = len(ENCODING.encode(conversation.convert_to_prompt()))
    if prompt_len + TOKEN_BUFFER < context_length:
        return conversation

    turns = conversation.turns
    if len(turns) == 1:
        # TODO: implement logic to cut down the last turn if it's too long
        return conversation

    tokens_per_turn = [len(ENCODING.encode(turn.content)) for turn in turns]
    # keep system turn if system turn is not over context length and there's more than 1 other turns
    if (
        turns[0].role == "system"
        and tokens_per_turn[0] + TOKEN_BUFFER <= context_length
        and len(turns) != 2
    ):
        return shorten_conversation(
            Conversation(turns=[turns[0]] + turns[2:]), context_length
        )
    else:
        return shorten_conversation(Conversation(turns=turns[1:]), context_length)


def _gen_single_answer(answer: Answer, question: Question) -> Optional[Answer]:
    """
    Generate one answer for a given answer object and fill in the content and cost fields.
    """
    adapter = AdapterFactory.get_adapter(answer.llm_id)

    conversation = shorten_conversation(
        question.conversation, adapter.get_context_length()
    )
    conversation = make_prompt_modification(conversation, answer.prompt_modification)

    # generate answer
    for _ in range(NUM_RETRIES):
        try:
            response = adapter.execute_sync(conversation, **answer.params)
            answer.content = (
                response.response.content
                if hasattr(response.response, "content")
                else response.response
            )
            answer.cost = response.cost
            return answer.save()
        except Exception as e:
            print(f"Error generating {question.id} for {answer.llm_id}: {e}")
            print(traceback.format_exc())


def gen_answers(
    answer_worker_inputs: List[Answer],
    answer_outputs: List[Answer],
    num_workers: int = 1,
) -> List[Answer]:
    """
    Generate answers for a list of answer objects in parallel using a ProcessPoolExecutor.
    """
    # Get all the questions at once from the answers
    answer_questions = Question.get_questions_by_id(
        [answer.question_id for answer in answer_worker_inputs]
    )
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for answer in tqdm(
            executor.map(_gen_single_answer, answer_worker_inputs, answer_questions),
            total=len(answer_worker_inputs),
            desc="Generating answers or ground truth answers",
        ):
            if answer:
                answer_outputs.append(answer)
    return answer_outputs


def _prepare_answer(
    answer: Answer, N: Optional[int] = 1
) -> Tuple[List[Answer], List[Answer]]:
    answer_outputs = []
    answer_worker_inputs = []
    existing_answers = get_existing_answers(answer, N)
    if existing_answers:
        answer_outputs.extend(existing_answers)
        num_additional_answers = N - len(existing_answers)
    else:
        num_additional_answers = N

    if num_additional_answers > 0:
        answer_worker_inputs.extend(
            [copy.deepcopy(answer) for _ in range(num_additional_answers)]
        )
    return answer_outputs, answer_worker_inputs


def prepare_answers(
    answers: List[Answer], N: Optional[int] = 1, num_workers: int = 1
) -> Tuple[List[Answer], List[Answer]]:
    """
    Find existing answers for each answer object and prepare the list of answers to generate.
    Return the list of existing answers and the list of answer objects to generate.
    """
    answer_outputs = []
    answer_worker_inputs = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for answer in tqdm(
            executor.map(_prepare_answer, answers, [N] * len(answers)),
            total=len(answers),
            desc="Preparing Answers",
        ):
            answer_outputs.extend(answer[0])
            answer_worker_inputs.extend(answer[1])

    print(f"Generating {len(answer_worker_inputs)} new answers")
    return answer_outputs, answer_worker_inputs


def main(
    output_dir: str,
    question_ids: List[str],
    llm_params: LLMParamsList,
    N: Optional[int] = 1,
    num_workers: int = 1,
    printout_limit: int = 100,
    seed: int = 42,
):
    """
    This is the main function to generate answers.
    Generate answers for a list of questions using the given llm parameters.
    """
    answers = create_answer_objects(question_ids, llm_params)
    answer_outputs, answer_worker_inputs = prepare_answers(
        answers, N, num_workers=num_workers
    )
    answer_outputs = gen_answers(answer_worker_inputs, answer_outputs, num_workers)

    random.seed(seed)
    random.shuffle(answer_outputs)
    write_ids_to_json(answer_outputs, os.path.join(output_dir, "answer_ids.json"))
    write_to_csv(
        answer_outputs[:printout_limit],
        os.path.join(output_dir, "sample_answers.csv"),  # 24-05-31_answers.csv
    )


def get_question_ids(output_dir: str) -> List[str]:
    with open(os.path.join(output_dir, "question_ids.json"), "r") as f:
        return json.load(f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--llm_params", type=LLMParamsList, required=True)
    parser.add_argument("--printout_limit", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--N", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", action=ActionConfigFile)

    args = parser.parse_args()
    save_experiment_config(
        os.path.join(args.output_dir, "gen_answers_config.yaml"),
        parser.dump(args, skip_none=True),
    )
    question_ids = load_from_json(os.path.join(args.output_dir, "question_ids.json"))

    main(
        output_dir=args.output_dir,
        question_ids=question_ids,
        llm_params=args.llm_params,
        N=args.N,
        num_workers=args.num_workers,
        printout_limit=args.printout_limit,
        seed=args.seed,
    )
