import os
import dspy
import pandas as pd
from dspy import OpenAI
from dspy.teleprompt import BootstrapFewShot, LabeledFewShot
from dspy.evaluate.evaluate import Evaluate
from llm_judge.database.models.prompts import Prompt
from llm_judge.database.models.questions import Question
from llm_judge.database.models.answers import Answer
from llm_judge.database.models.judgments import Judgment
from llm_judge.database.models.eval_results import EvalResult
from dotenv import load_dotenv

load_dotenv(".env")

OAI_API_KEY_ID = "OPENAI_API_KEY"


class JudgementSignature(dspy.Signature):
    """Determine which one is better. Think step by step and explain your reasoning. Then, decide if the candidate answer is as good as the reference answer.
    Avoid any position biases and ensure that the order in which the candidate and reference answers were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible.
    After you output your analysis on the criteria, output your final verdict at the end by **strictly following the following format**:
    ```Final Verdict: "[[A]]"``` if candidate answer is better than the reference answer.
    ```Final Verdict: "[[B]]"``` if the candidate is worse than the reference answer.
    ```Final Verdict: "[[C]]"``` if they are similar in quality.
    """

    user_prompt = dspy.InputField(
        desc="The prompt which contain user conversation and both candidate and reference answer"
    )
    output = dspy.OutputField(desc="The output verdict")


class Judgementor(dspy.Module):
    """
    A dspy module to initialize the OpenAI model, initialize dspy.Predict module and predict the judgement based on the given user prompt.
    """

    def __init__(self, model_id: str = "gpt-3.5-turbo-0125"):
        """
        Initialisation of the dspy module class

        Args:
            model_id (str): The ID of the language model to use (default: "gpt-3.5-turbo-0125").
        """

        # Initialize the language model with the given model_id and API key
        global llm
        llm = OpenAI(
            model=model_id, max_tokens=2000, api_key=os.environ.get(OAI_API_KEY_ID)
        )
        dspy.settings.configure(lm=llm)

        # Set up the prediction pipeline with the defined signature
        self.predictor = dspy.Predict(JudgementSignature)

    def forward(self, user_prompt):
        """
        Generate a judgement based on the user prompt.

        Args:
            user_prompt (str): The prompt containing the user conversation and both candidate and reference answers.

        Returns:
            dspy.Prediction: The prediction details including the judgement, used prompt, and model response.
        """

        # Make a prediction object based on the user prompt
        prediction = self.predictor(user_prompt=user_prompt)
        # print("Prediction: \n", prediction)

        # # Retrieve the last used prompt and the model's response from the history
        global llm
        prompt_used = llm.history[-1][
            "prompt"
        ]  # inspect_history(n=1).split("Output")[0]
        response = llm.history[-1]["response"]

        # Return the prediction details
        return dspy.Prediction(
            prompt=user_prompt,
            judgement=prediction.output,
            updated_prompt=prompt_used,
            response=response,
        )


class PromptTuner:
    """
    Main class for automatically updating the judgement prompt.

    This class prepares training data, implements the DSPy Signature and module, trains the model using DSPy compiler, and evaluates its performance against the ideal judgement.

    """

    def __init__(
        self,
        model_id="gpt-3.5-turbo",
        train_df_path="./enriched_judgments.csv",
        test_size=0.1,
    ):
        """
        Initialisation of the the class

        Args:
            model_id (str): The ID of the language model to use (default: "gpt-3.5-turbo").
            train_df_path (str): The path to the training data CSV file.
            test_size (float): The proportion of data to be used for testing (default: 0.1).
        """

        self.train_df_path = train_df_path
        self.model_id = model_id
        self.test_size = test_size

        data = self.prepare_training_data()
        test_len = int(test_size * len(data))
        self.train_data = data[test_len:]
        self.eval_data = data[:test_len]

    @staticmethod
    def check_verdict(judgement_out):
        """
        Check the verdict in the judgement output.

        Args:
            judgement_out (str): The output of the judgement.

        Returns:
            str: 'A', 'B', 'C', or 'error' based on the judgement output.
        """

        if "[[A]]" in judgement_out:
            return "A"
        elif "[[B]]" in judgement_out:
            return "B"
        elif "[[C]]" in judgement_out:
            return "C"
        else:
            return "error"

    def validate_judgements(
        self, example: dspy.Example, pred: dspy.Prediction, trace=None
    ):
        """
        metrics function used by the compiler the to comparing example output and predicted output.

        Args:
            example (dspy.Example): The example containing the user prompt and expected output.
            pred (dspy.Prediction): The predicted judgement.
            trace (optional): Additional trace information.

        Returns:
            bool: True if example verdict matches predicted verdict, otherwise False.
        """

        # print("Example Output: \n",example.output)

        example_verdict = PromptTuner.check_verdict(example.output)
        pred_verdict = PromptTuner.check_verdict(pred.judgement)
        # print(example_verdict, pred_verdict)

        eval_result = EvalResult(
            judge_prompt=example.user_prompt,
            judge_score=float(example_verdict == pred_verdict),
            args=dict(prompt=pred.updated_prompt, response=pred.response),
        )
        eval_result.save()

        return example_verdict == pred_verdict

    def prepare_training_data(self):
        """
        Prepare training data from the enriched judgements CSV file.

        Returns:
            list: A list of dspy.Example objects for training.
        """

        training_data = []
        enrich_judgments_df = pd.read_csv(self.train_df_path)
        for _, row in enrich_judgments_df.iterrows():
            judgement = Judgment.get_judgment_by_id(row["id"])
            judge_prompt = Prompt.get_prompt_by_id("662821e23eb9ef01018e30e2").content
            question = Question.get_question_by_id(judgement.question_id)
            answer = Answer.get_answer_by_id(judgement.answer_id)
            reference_answer = Answer.get_answer_by_id(
                judgement.comparison_answers[0]
            ).content
            prompt = judge_prompt.format(
                question=question.conversation.convert_to_prompt(),
                reference=reference_answer,
                candidate=answer.content,
            )
            output = judgement.judgments[0]
            example = dspy.Example(user_prompt=prompt, output=output).with_inputs(
                "user_prompt"
            )
            training_data.append(example)
        print("No. of training data: ", len(training_data))
        return training_data

    @classmethod
    def load(cls, path):
        """
        Load a trained Judgementor model from the given path.

        Args:
            path (str): The path to the saved model.

        Returns:
            Judgementor: The loaded Judgementor model.
        """
        judge_mentor = Judgementor()
        return judge_mentor.load(path)

    def train(self, save_path):
        """
        Train the Judgementor model using the training data.

        Args:
            save_path (str): The path to save the compiled model.
        """

        # Set up a basic teleprompter, which will compile our RAG program.
        teleprompter = BootstrapFewShot(
            metric=self.validate_judgements,
            max_bootstrapped_demos=4,
            max_labeled_demos=3,
        )
        # teleprompter2 = LabeledFewShot(k=2)

        # Compile the DSPy module
        self.compiled_class = teleprompter.compile(
            Judgementor(model_id=self.model_id), trainset=self.train_data
        )
        self.compiled_class.save(save_path)

    def eval(self):
        """
        Evaluate the trained Judgementor model using the evaluation data.

        Returns:
            float: which contain the accuracy score of the evaluation set
        """
        evaluator = Evaluate(
            devset=self.eval_data, num_threads=1, display_progress=True, display_table=5
        )
        styled_df = evaluator(self.compiled_class, metric=self.validate_judgements)
        return styled_df


if __name__ == "__main__":

    prompt_tuner = PromptTuner(
        model_id="gpt-3.5-turbo-0125",
        train_df_path="/home/anon/freelancing/mercor_mle/suresh-llm-judge-interview-2/llm-judge-interview-2/llm-judge-interview-2/data/mercor_mle/test-2024-05-17-00-29/enriched_judgments.csv",
        test_size=0.1,
    )
    prompt_tuner.train("dspy_compiled.json")
    print(f"Accuracy of the compiled model: {prompt_tuner.eval()}")
