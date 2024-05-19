"""
given a ground truth, evaluate quailty of the output of a judge.
"""

import yaml
from llm_judge.database.models.eval_result import EvalResult
from llm_judge.judges import (
    BaselineComparisonJudge,
    GroundTruthComparisonJudge,
    ExactMatchJudge,
)
import os.path
import pandas as pd
from datetime import datetime
from typing import Literal
from llm_judge.database.models.prompts import Prompt
import yaml
import subprocess
import ast
import shutil
import json
import random

from scripts.gen_judgments import main as gen_judgments
from llm_judge.judges import BaselineComparisonJudge
from llm_judge.judges.concrete_judges.dynamic_few_shot_judge import (
    DynamicFewShotJudge,
)  # expecting voyager API key when importing because of the pkg init but solved by putting client creation into class
from llm_judge.judges.abstract_judges.judge import Judge
from llm_judge.utils.common import load_from_json, write_df_to_csv


class EvalJudge:
    """
    Evaluate the quality of a judge prompt by comparing it to a ground truth.
    """

    def __init__(
        self,
        ideal_judgment_path: str,
        judge_class: Judge = BaselineComparisonJudge,
        experiments_run_folder: str = "../data/auto_judge/",
        raw_data_folder: str = "data/DeepAI/test-2024-04-30-17-52",
        random_seed: int = 42,
        test_split: float = 0.9,
        judgments_for_few_shot_retrieval_fp: str = None,
    ):

        # set seed
        random.seed(random_seed)
        self.judge_class = judge_class
        self.datetime_str = datetime.now().strftime("%m-%d_%H-%M-%S")
        self.raw_data_folder = raw_data_folder

        self.ideal_judgment_df = pd.read_csv(ideal_judgment_path)
        self.ideal_judgment_df["ideal_judgment_score"] = self.ideal_judgment_df[
            "score"
        ].copy()
        self.ideal_judgment_df.drop(columns=["score"], inplace=True)

        self.train_df = self.ideal_judgment_df.sample(frac=1 - test_split)
        self.test_df = self.ideal_judgment_df.drop(self.train_df.index)
        self.output_dir = os.path.join(experiments_run_folder, self.datetime_str)
        os.makedirs(self.output_dir, exist_ok=True)

        # Copy gen_questions_config.yaml and gen_answers_config.yaml from raw_data_folder to output_dir
        shutil.copy(
            os.path.join(self.raw_data_folder, "gen_questions_config.yaml"),
            self.output_dir,
        )
        shutil.copy(
            os.path.join(self.raw_data_folder, "gen_answers_config.yaml"),
            self.output_dir,
        )

        # assert "ideal_judgment_score" in self.ideal_judgment_df.columns
        # assert "ideal_judgment_rationale" in self.ideal_judgment_df.columns

        self.config_template = {
            "judge": {
                "init_args": {
                    "baseline_llm": "gpt-3.5-turbo-0125",
                    "judge_prompts": {
                        "system": None,
                        "prompt_template": "662821e23eb9ef01018e30e2",
                        "reversed_prompt_template": "6628224eb84c0693351ca6a4",
                    },
                    "judge_llm": "gpt-4-turbo-preview",
                    "judge_llm_params": {"temperature": 0, "max_tokens": 3000},
                },
            },
            "N": 1,
            "output_dir": self.output_dir,
            "num_workers": 30,
            "printout_limit": 500,
            "output_enriched": True,
        }

        if self.judge_class == DynamicFewShotJudge:
            assert (
                judgments_for_few_shot_retrieval_fp
            ), "judgments_for_few_shot_retrieval_fp must be provided for DynamicFewShotJudge"
            self.config_template["judge"]["init_args"][
                "ideal_judgment_path"
            ] = judgments_for_few_shot_retrieval_fp

    def evaluate_judge_prompt(
        self, judge_prompt: str, eval_on=Literal["train", "test", "all"]
    ) -> pd.DataFrame:
        """
        Evaluate the quality of a judge prompt by comparing it to a ground truth.
        """
        match eval_on:
            case "train":
                eval_df = self.train_df
            case "test":
                eval_df = self.test_df
            case "all":
                eval_df = self.ideal_judgment_df
            case _:
                raise ValueError("eval_data must be 'train', 'test', or 'all'")

        self._export_all_ids(eval_df, self.output_dir)

        exp_judge_results = self.run_judge_with_prompt(judge_prompt=judge_prompt)

        # to stimulate that we have create ideal_judge.csv
        # ideal_judge_results = self.run_judge_with_prompt(judge_prompt_id="662813e0e25b6076a9e03df8")
        # ideal_judge_results.drop(columns=["score"], inplace=True) # to let merge result only contain score that is from exp_prompt
        merged_results = pd.merge(
            exp_judge_results,
            self.ideal_judgment_df[["answer_id", "ideal_judgment_score"]],
            on="answer_id",
            how="left",
        )

        # store
        # ideal_jscore = self.ideal_judgment_df["score"]
        # exp_jscore = merged_results["exp_judge_score"]
        merged_results["exp_judge_score"] = exp_judge_results["score"].apply(
            ast.literal_eval
        )
        merged_results["ideal_judge_score"] = merged_results[
            "ideal_judgment_score"
        ].apply(ast.literal_eval)

        # acc_score = (merged_results["exp_judge_score"] == merged_results["ideal_judge_score"]).mean()

        # result = {
        #     "judge_score": acc_score,
        #     "judge_prompt": judge_prompt,
        #     "ideal_prompt_score": ideal_jscore,
        #     "exp_prompt_score": exp_jscore,
        #     "eval_on": eval_df
        # }

        return merged_results

    def run_judge_with_prompt(self, judge_prompt: str) -> pd.DataFrame:
        """
        Execute a judge prompt and return the results.
        """
        judge_prompt_id = self.prep_prompt(judge_prompt)
        configs = self.prep_config(judge_prompt_id)
        judge = self.judge_class(**configs["judge"]["init_args"])

        question_ids = load_from_json(
            os.path.join(configs["output_dir"], "question_ids.json")
        )
        answer_ids = load_from_json(
            os.path.join(configs["output_dir"], "answer_ids.json")
        )

        result_fp = gen_judgments(
            output_dir=configs["output_dir"],
            judge=judge,
            question_ids=question_ids,
            answer_ids=answer_ids,
            N=configs["N"],
            num_workers=configs["num_workers"],
            printout_limit=configs["printout_limit"],
            output_enriched=configs["output_enriched"],
        )
        print("the ↑ above result is experimental judge results")
        enriched_judge_results = pd.read_csv(result_fp)
        return enriched_judge_results

    def prep_prompt(self, judge_prompt: str) -> str:
        judge_prompt_base = """
Think step by step and explain your reasoning. Then, decide if the candidate answer is as good as the reference answer.
Avoid any position biases and ensure that the order in which the candidate and reference answers were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible.
After you output your analysis on the criteria, output your final verdict at the end by **strictly following the following format**:
```Final Verdict: "[[A]]"``` if candidate answer is better than the reference answer.
```Final Verdict: "[[B]]"``` if the candidate is worse than the reference answer.
```Final Verdict: "[[C]]"``` if they are similar in quality.
"""
        judge_prompt_obj = Prompt(
            description="test judge prompt in eval_judge.py",
            content=judge_prompt + judge_prompt_base,
            tags=["test", "baseline_pairwise", "reversed_prompt_template"],
        )
        judge_prompt_obj = judge_prompt_obj.get_or_save()
        judge_prompt_id = judge_prompt_obj.id
        return str(judge_prompt_id)

    def prep_config(self, judge_prompt_id: str) -> dict:
        config_data = self.config_template.copy()
        config_data["judge"]["init_args"]["judge_prompts"]["system"] = str(
            judge_prompt_id
        )
        return config_data

    def _export_all_ids(self, judgment_df_to_eval: pd.DataFrame, output_folder: str):
        judgment_df_to_eval.question_id.to_json(
            os.path.join(output_folder, "question_ids.json"), orient="values"
        )
        ansewr_ids = judgment_df_to_eval.answer_id.to_list()
        ansewr_ids += judgment_df_to_eval.comparison_answers.apply(
            lambda s: s.replace("[ObjectId('", "").replace("')]", "")
        ).to_list()
        with open(os.path.join(output_folder, "answer_ids.json"), "w") as file:
            json.dump(ansewr_ids, file)


if __name__ == "__main__":

    # Load configuration from config.yaml
    with open("eval.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Map the judge class from string to actual class
    judge_class_mapping = {
        "BaselineComparisonJudge": BaselineComparisonJudge,
        "GroundTruthComparisonJudge": GroundTruthComparisonJudge,
        "ExactMatchJudge": ExactMatchJudge,
    }

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

        write_df_to_csv(eval_result, f"{judge_prompt[:10]-{i}}")

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
