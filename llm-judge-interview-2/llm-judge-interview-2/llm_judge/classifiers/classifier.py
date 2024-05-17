from abc import ABC, abstractmethod
from llm_judge.database.models.questions import Question


class Classifier(ABC):
    @abstractmethod
    def generate_classification(self, question: Question) -> str:
        """
        Generate a classification for the question
        """
        raise NotImplementedError

    @abstractmethod
    def get_class_names(self) -> list[str]:
        """
        Get the class names for the classifier. Ex. ["positive", "negative"]
        """
        raise NotImplementedError

    def store_classification(self, question: Question, classification: str) -> Question:
        """
        Store the classification in the question object
        For now, we store classes in a list inside the args field of Question
        """
        if "classes" in question.args.keys():
            question.args["classes"].append(classification)
        else:
            question.args["classes"] = [classification]

        return question

    def classify(self, question: Question) -> Question:
        """
        add classification to the args field of the question
        """
        classification = self.generate_classification(question)
        return self.store_classification(question, classification)
