import os
import json
from tqdm import tqdm
from datetime import datetime
import traceback
import random
from concurrent.futures import ProcessPoolExecutor
from jsonargparse import ActionConfigFile, ArgumentParser
from typing import Optional, List

from llm_judge.database.models.questions import Question
from llm_judge.question_generator.convo_json_question_generator import (
    BaseQuestionGenerator,
)
from llm_judge.utils.common import (
    write_to_csv,
    write_ids_to_json,
    combine_and_deduplicate,
    save_experiment_config,
)
from llm_judge.classifiers.classifier import Classifier
from llm_judge.utils.types import LLMParamsList
from llm_judge.utils.ground_truth import gen_gronud_truth


def batch_classify(
    questions: List[Question], classifier: Classifier, num_workers: int = 1
):
    """
    Classify a list of questions in parallel using the given classifier.
    """
    questions_to_classify = []
    classified_questions = []
    for q in questions:
        if q.is_classified(classifier.get_class_names()):
            classified_questions.append(q)
        else:
            questions_to_classify.append(q)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_question = {
            executor.submit(classifier.classify, question): question
            for question in questions_to_classify
        }
        for future in tqdm(
            future_to_question,
            total=len(future_to_question),
            desc="Classifying questions",
        ):
            try:
                classified_questions.append(future.result().update())
            except Exception as e:
                question = future_to_question[future]
                print(f"Error classifying question {question.id}: {e}")
                traceback.print_exc()
                classified_questions.append(question)
    return classified_questions


def load_existing_questions(file_path: str) -> List[Question]:
    """
    Load existing questions from a json file.
    Assume the json file contains a list of question ids.
    """
    with open(file_path, "r") as f:
        question_ids = json.load(f)
    return Question.get_questions_by_id(question_ids)


def main(
    output_dir: str,
    existing_question_path: Optional[str] = None,
    question_generator: Optional[BaseQuestionGenerator] = None,
    wanted_classes: List[str] = [],
    ground_truth_params: LLMParamsList = [],
    ground_truth_aggregator_params: LLMParamsList = [
        {"gpt-4-turbo-preview": {"temperature": 0}}
    ],
    classifiers: Optional[List[Classifier]] = [],
    printout_limit: int = 100,
    seed: int = 42,
    num_workers: int = 1,
):
    """
    This is the main function to generate questions.
    Generate questions, classify them, generate ground truth, filter wanted questions, and save the questions.
    """
    questions_from_data_files = Question.batch_update_or_save(
        question_generator.make_questions()
    )
    existing_questions = (
        load_existing_questions(existing_question_path)
        if existing_question_path
        else []
    )
    if len(existing_questions) > 0:
        questions = combine_and_deduplicate(
            questions_from_data_files, existing_questions
        )
    else:
        questions = questions_from_data_files

    for classifier in classifiers:
        questions = batch_classify(questions, classifier, num_workers)

    if wanted_classes:
        for class_name in wanted_classes:
            questions = [
                question for question in questions if question.is_class(class_name)
            ]

    if not questions:
        print("No questions were present in wanted classes.")
        return

    if ground_truth_params:
        questions = gen_gronud_truth(
            ground_truth_params, ground_truth_aggregator_params, questions, num_workers
        )

    random.seed(seed)
    random.shuffle(questions)
    write_ids_to_json(questions, os.path.join(output_dir, "question_ids.json"))
    write_to_csv(
        questions[:printout_limit], os.path.join(output_dir, "sample_questions.csv")
    )


def validate_and_process_args(args):
    """
    Validate the arguments collected by argparse.
    """
    if args.base_dir is not None:
        assert (
            args.output_dir is None
        ), "Output dir is given to update an existing run. Only one of base_dir and output_dir can be given."
        datetime_str = datetime.now().strftime("%Y-%m-%d-%H-%M")
        args.output_dir = os.path.join(
            args.base_dir, f"{args.experiment_name}-{datetime_str}"
        )
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        assert args.output_dir is not None and os.path.exists(
            args.output_dir
        ), "Either base_dir or output_dir must be given. Output dir must exist."

    return args


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--question_generator", type=BaseQuestionGenerator, required=True
    )
    parser.add_argument(
        "--classifiers", type=List[Classifier], required=False, default=[]
    )
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, required=False, default="")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--wanted_classes", type=List[str], required=False, default=[])
    parser.add_argument(
        "--ground_truth_params", type=LLMParamsList, required=False, default=[]
    )
    parser.add_argument(
        "--ground_truth_aggregator_params",
        type=LLMParamsList,
        required=False,
        default=[{"gpt-4-turbo-preview": {"temperature": 0}}],
    )
    parser.add_argument(
        "--existing_question_path", type=str, required=False, default=None
    )
    parser.add_argument("--printout_limit", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", action=ActionConfigFile)

    args = parser.parse_args()
    validate_and_process_args(args)
    save_experiment_config(
        os.path.join(args.output_dir, "gen_questions_config.yaml"),
        parser.dump(args, skip_none=True),
    )

    args = parser.instantiate_classes(args)
    main(
        output_dir=args.output_dir,
        existing_question_path=args.existing_question_path,
        question_generator=args.question_generator,
        wanted_classes=args.wanted_classes,
        ground_truth_params=args.ground_truth_params,
        ground_truth_aggregator_params=args.ground_truth_aggregator_params,
        classifiers=args.classifiers,
        printout_limit=args.printout_limit,
        seed=args.seed,
        num_workers=args.num_workers,
    )
