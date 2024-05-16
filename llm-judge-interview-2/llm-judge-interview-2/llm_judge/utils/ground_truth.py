from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
from tqdm import tqdm
from collections import defaultdict
from adapters.types import Conversation, Turn, ConversationRole
from adapters.adapter_factory import AdapterFactory

from llm_judge.database.models.answers import Answer
from llm_judge.database.models.questions import Question
from scripts.gen_answers import create_answer_objects, gen_answers, prepare_answers
from llm_judge.utils.types import LLMParamsList
from llm_judge.utils.common import gen_hash, stringify_llm_params


EQUIVALENCE_COMPARISON_TEMPLATE = """Please act impartially and judge the following two answers based on how similar they are to each other and if they arrive at the same answer. Explain your reasoning. If both answers are functionally the same, please respond with [[A]]. If both answers are functionally different, please respond with [[B]]. If both answers arrive at the same solution, even if by different means, please respond with [[A]]. After providing your explanation, provide the rating of "A" or "B" in your response, in the format [[X]], where X is the rating.\n\n[Answer 1]\n{answer1}\n\n[Answer 2]\n{answer2}"""


def gen_ground_truth_hash(
    ground_truth_params: LLMParamsList, ground_truth_aggregator_params: LLMParamsList
) -> str:
    """
    Generate a combined hash for the ground truth params and aggregator params.
    """
    params_hash = gen_hash(stringify_llm_params(ground_truth_params))
    aggregator_hash = gen_hash(stringify_llm_params(ground_truth_aggregator_params))
    return gen_hash(params_hash + aggregator_hash)


def filter_questions(questions: List[Question], ground_truth_hash: str):
    """Separate questions into those needing ground truth generation and those already having it."""
    to_generate = []
    with_ground_truth = []
    for question in questions:
        if ground_truth_hash not in question.ground_truth:
            to_generate.append(question)
        else:
            with_ground_truth.append(question)
    return to_generate, with_ground_truth


def group_answers(answers: List[Answer]) -> Dict[str, List[Answer]]:
    """
    Group answers by question id.
    """
    answers_per_question = {}
    for answer in answers:
        if answer.question_id not in answers_per_question:
            answers_per_question[answer.question_id] = [answer]
        answers_per_question[answer.question_id].append(answer)
    return answers_per_question


def compare_answer_contents(
    answer1: str, answer2: str, aggregator_params: LLMParamsList
) -> bool:
    """
    Compare two answers to determine if they are functionally equivalent
    """
    assert len(LLMParamsList) == 1
    aggregate_llm_params = aggregator_params[0]
    llm_id = list(aggregate_llm_params.keys())[0]
    params = aggregate_llm_params[llm_id]
    conversation = Conversation(
        [
            Turn(
                role=ConversationRole.user,
                content=EQUIVALENCE_COMPARISON_TEMPLATE.format(
                    answer1=answer1, answer2=answer2
                ),
            ),
        ]
    )
    adapter = AdapterFactory.get_adapter(llm_id)
    try:
        response = adapter.execute_sync(conversation, **params)
        return (
            response.content.split("[[")[1].split("]]")[0] == "A"
        )  # this is custom to the comparison prompt
    except Exception as e:
        print(f"Error generating ground truth: {e}")
        print(traceback.format_exc())
        return False


def aggregate_answers(
    answers: List[Answer], ground_truth_aggregator_params: LLMParamsList
) -> str:
    """
    This function compares the answers and returns the majority voted ground truth, or the highest voted answer if there is no majority
    """
    if not answers:
        return ""
    majority_threshold = (len(answers) + 1) // 2
    count = defaultdict(int)
    for answer in answers:
        count[answer.content] += 1

    answer_contents = list(count.keys())
    for i, answer1 in enumerate(answer_contents):
        if count[answer1] >= majority_threshold:
            return answer1

        for answer2 in answer_contents[i + 1 :]:
            if compare_answer_contents(
                answer1, answer2, ground_truth_aggregator_params
            ):
                count[answer1] += 1
                count[answer2] += 1
                if count[answer1] >= majority_threshold:
                    return answer1
                if count[answer2] >= majority_threshold:
                    return answer2

    return max(count, key=lambda k: count[k])


def generate_and_group_ground_truth_answers(
    questions_to_generate: List[Question],
    ground_truth_params: LLMParamsList,
    num_workers: int = 1,
) -> Dict[str, List[Answer]]:
    """Generate answers for the given questions using the ground truth params and group them by question."""
    answer_templates = create_answer_objects(
        [question.id for question in questions_to_generate], ground_truth_params
    )
    answer_outputs, answer_worker_inputs = prepare_answers(answer_templates, 1)
    answer_outputs = gen_answers(answer_worker_inputs, answer_outputs, num_workers)
    return group_answers(answer_outputs)


def aggregate_and_save_ground_truth(
    ground_truth_hash: str,
    answers_per_question: Dict[str, List[Answer]],
    ground_truth_aggregator_params: LLMParamsList,
    num_workers: int = 1,
) -> List[Question]:
    """
    Aggregate (ex. majority vote) answers for each question and save the ground truth.
    """
    updated_questions = []
    future_to_question = {}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for question_id, answers in answers_per_question.items():
            future = executor.submit(
                aggregate_answers, answers, ground_truth_aggregator_params
            )
            future_to_question[future] = question_id

        for future in tqdm(
            as_completed(future_to_question),
            total=len(future_to_question),
            desc="Aggregating Answers",
        ):
            question_id = future_to_question[future]
            aggregated_result = future.result()
            question = Question.get_question_by_id(question_id)
            question.ground_truth[ground_truth_hash] = aggregated_result
            updated_questions.append(question.update())

    return updated_questions


def gen_gronud_truth(
    ground_truth_params: LLMParamsList,
    ground_truth_aggregator_params: LLMParamsList,
    questions: List[Question],
    num_workers: int = 1,
) -> List[Question]:
    ground_truth_hash = gen_ground_truth_hash(
        ground_truth_params, ground_truth_aggregator_params
    )
    questions_to_generate, questions_with_ground_truth = filter_questions(
        questions, ground_truth_hash
    )
    answers_per_question = generate_and_group_ground_truth_answers(
        questions_to_generate, ground_truth_params, num_workers
    )
    updated_questions = aggregate_and_save_ground_truth(
        ground_truth_hash,
        answers_per_question,
        ground_truth_aggregator_params,
        num_workers,
    )
    questions = questions_with_ground_truth + updated_questions
    return questions
