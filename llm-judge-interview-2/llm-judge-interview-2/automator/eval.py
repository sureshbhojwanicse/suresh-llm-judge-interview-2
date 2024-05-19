from eval_judge import EvalJudge
import yaml
from llm_judge.database.models.eval_result import EvalResult
from llm_judge.judges import (
    BaselineComparisonJudge,
    GroundTruthComparisonJudge,
    ExactMatchJudge,
)

# Load configuration from config.yaml
with open("eval.yaml", "r") as file:
    config = yaml.safe_load(file)

# Map the judge class from string to actual class
judge_class_mapping = {
    "BaselineComparisonJudge": BaselineComparisonJudge,
    "GroundTruthComparisonJudge": GroundTruthComparisonJudge,
    "ExactMatchJudge": ExactMatchJudge,
}

if __name__ == "__main__":

    # Initialize the EvalJudge with the provided configurations
    eval_judge = EvalJudge(
        ideal_judgment_path=config["eval_judge"]["ideal_judgment_path"],
        judge_class=judge_class_mapping[config["eval_judge"]["judge_class"]],
        raw_data_folder=config["eval_judge"]["raw_data_folder"],
        test_split=config["eval_judge"]["test_split"],
    )

    # Get prompts from configuration
    prompts = config["prompts"]

    # Evaluate each prompt
    for i, judge_prompt in enumerate(prompts, start=1):
        print(f"========\n== {i}  ==\n========")
        # Evaluate the judge prompt and get the evaluation result
        eval_result = eval_judge.evaluate_judge_prompt(judge_prompt, config["eval_on"])

        # Calculate accuracy score by comparing experimental and ideal judge scores
        acc_score = (
            eval_result["exp_judge_score"] == eval_result["ideal_judge_score"]
        ).mean()

        # Create an instance of EvalResult and save it to MongoDB
        result = EvalResult(
            judge_prompt=judge_prompt,
            judge_score=acc_score,
        )
        result.save()
