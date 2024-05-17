import os
from typing import List, Optional, Tuple, Optional
import random
import copy
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from tqdm import tqdm
from jsonargparse import ActionConfigFile, ArgumentParser

from llm_judge.judges.abstract_judges.judge import Judge
from llm_judge.database.models.judgments import Judgment
from llm_judge.utils.types import JudgeType
from llm_judge.database.models.questions import Question
from llm_judge.database.models.answers import Answer
from llm_judge.utils.common import (
    load_from_json,
    write_ids_to_json,
    write_to_csv,
    write_df_to_csv,
    save_experiment_config,
)
from llm_judge.utils.ground_truth import gen_ground_truth_hash
from llm_judge.utils.types import convert_to_llm_params_list
from llm_judge.utils.common import parse_yaml_file


def create_judgment_objects(
    questions: List[Question],
    answers: List[Answer],
    judge: Judge,
    ground_truth_hash: str,
) -> List[Judgment]:
    """
    Create judgment objects for a list of questions and answers using the given judge.
    """
    judge_type = judge.get_type()
    if judge_type in [JudgeType.GENERATIVE_GROUND_TRUTH, JudgeType.EXACT_MATCH]:
        assert (
            ground_truth_hash is not None
        ), "Ground truth was not generated during gen_questions"
        for question in questions:
            assert (
                ground_truth_hash in question.ground_truth
            ), f"Ground truth with hash = {ground_truth_hash} was not generated for question_id = {question.id}"
    return judge.make_pairs(questions, answers)


def _prepare_judgments(
    judgment: Judgment, N: int
) -> Tuple[List[Judgment], List[Judgment]]:
    """
    Find existing judgments for each judgment object and prepare the list of judgments to generate.
    Return the list of existing judgments and the list of judgment objects to generate.
    """
    judgment_outputs = []
    judgment_worker_inputs = []

    existing_judgments = judgment.get_judgment_by_args()
    if existing_judgments:
        judgment_outputs.extend(existing_judgments)
        num_additional_judgments = N - len(existing_judgments)
    else:
        num_additional_judgments = N

    if num_additional_judgments > 0:
        judgment_worker_inputs.extend(
            [copy.deepcopy(judgment) for _ in range(num_additional_judgments)]
        )

    return judgment_outputs, judgment_worker_inputs


def prepare_judgments(
    judgments: List[Judgment], N: int, num_workers: int = 30
) -> Tuple[List[Judgment], List[Judgment]]:
    judgment_outputs = []
    judgment_worker_inputs = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for judgment in tqdm(
            executor.map(
                _prepare_judgments,
                judgments,
                [N] * len(judgments),
            ),
            total=len(judgments),
            desc="Preparing Judgments",
        ):
            judgment_outputs.extend(judgment[0])
            judgment_worker_inputs.extend(judgment[1])

    print(f"Generating {len(judgment_worker_inputs)} new judgments")

    return judgment_outputs, judgment_worker_inputs


def gen_judgments(
    judge: Judge,
    judgment_worker_inputs: List[Judgment],
    judgment_outputs: List[Judgment],
    num_workers: int,
) -> List[Judgment]:

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for answer in tqdm(
            executor.map(judge.generate_single_judgment, judgment_worker_inputs),
            total=len(judgment_worker_inputs),
            desc="Generating judgements",
        ):
            if answer:
                judgment_outputs.append(answer)
    return judgment_outputs


def get_ground_truth_hash(judge: Judge, yaml_file: str) -> Optional[str]:
    gen_questions_config = parse_yaml_file(yaml_file)
    if not (
        "ground_truth_params" in gen_questions_config
        and "ground_truth_aggregator_params" in gen_questions_config
    ):
        return None
    if judge.get_type() == JudgeType.EXACT_MATCH:
        return ""
    ground_truth_params = convert_to_llm_params_list(
        gen_questions_config["ground_truth_params"]
    )
    ground_truth_aggregator_params = convert_to_llm_params_list(
        gen_questions_config["ground_truth_aggregator_params"]
    )
    return gen_ground_truth_hash(ground_truth_params, ground_truth_aggregator_params)


def enrich_judgments(judgments: List[Judgment]) -> List[dict]:
    """
    Enrich the judgments with additional information.
    """
    enriched_judgments = []
    for judgment in tqdm(judgments, desc="Enriching Judgments"):
        enriched_judgment = judgment.model_dump()
        enriched_judgment["question"] = (
            Question.get_question_by_id(judgment.question_id)
            .conversation.convert_to_prompt()
            .__str__()
        )

        answer_obj = Answer.get_answer_by_id(judgment.answer_id)
        enriched_judgment["answer"] = answer_obj.content
        enriched_judgment["answer_llm"] = answer_obj.llm_id
        assert not (
            judgment.comparison_answers and judgment.ground_truth
        ), "only one of ground_truth or comparison_answers can exist"
        if judgment.comparison_answers:
            assert (
                len(judgment.comparison_answers) == 1
            ), "Multi-comparison not defined for enrich_judgments yet"
            gt_obj = Answer.get_answer_by_id(judgment.comparison_answers[0])
            enriched_judgment["reference_answer"] = gt_obj.content
            enriched_judgment["reference_llm"] = gt_obj.llm_id
        elif judgment.ground_truth:
            enriched_judgment["reference_answer"] = judgment.ground_truth
            enriched_judgment["reference_llm"] = "None"
        else:
            raise ValueError("No ground truth or comparison answers found")
        enriched_judgments.append(enriched_judgment.copy())
    return enriched_judgments


def main(
    output_dir: str,
    judge: Judge,
    question_ids: List[str],
    answer_ids: List[str],
    N: Optional[int] = 1,
    num_workers: int = 1,
    printout_limit: int = 100,
    seed: int = 42,
    output_enriched: bool = False,
    # write_ids_and_csv=True
) -> str:
    """
    This is the main function to generate judgments.
    Create judgments for a list of questions and answers, then use the given judge to generate judgments.
    """

    questions = Question.get_questions_by_id(question_ids)
    answers = Answer.get_answers_by_id(answer_ids)
    ground_truth_hash = get_ground_truth_hash(
        judge, os.path.join(output_dir, "gen_questions_config.yaml")
    )
    judgments = create_judgment_objects(questions, answers, judge, ground_truth_hash)
    judgment_outputs, judgment_worker_inputs = prepare_judgments(judgments, N)
    judgment_outputs = gen_judgments(
        judge, judgment_worker_inputs, judgment_outputs, num_workers
    )

    random.seed(seed)
    random.shuffle(judgment_outputs)
    # if write_ids_and_csv:
    write_ids_to_json(judgment_outputs, os.path.join(output_dir, "judgment_ids.json"))
    write_to_csv(
        judgment_outputs[:printout_limit],
        os.path.join(output_dir, "sample_judgments.csv"),
    )
    if output_enriched:
        enriched_output_path = os.path.join(
            output_dir, "enriched_judgments.csv"
        )  # Changed here
        enriched_judgments = enrich_judgments(judgment_outputs)
        write_df_to_csv(
            enriched_judgments[:printout_limit],
            enriched_output_path,
        )
        return enriched_output_path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--judge", type=Judge, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--N", type=int, default=1)
    parser.add_argument("--printout_limit", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument("--output_enriched", type=bool, default=True)

    args = parser.parse_args()
    save_experiment_config(
        os.path.join(args.output_dir, "gen_judgments_config.yaml"),
        parser.dump(args, skip_none=True),
    )
    args = parser.instantiate_classes(args)

    question_ids = load_from_json(os.path.join(args.output_dir, "question_ids.json"))
    answer_ids = load_from_json(os.path.join(args.output_dir, "answer_ids.json"))
    main(
        args.output_dir,
        args.judge,
        question_ids,
        answer_ids,
        args.N,
        args.num_workers,
        args.printout_limit,
        args.seed,
        args.output_enriched,
    )
