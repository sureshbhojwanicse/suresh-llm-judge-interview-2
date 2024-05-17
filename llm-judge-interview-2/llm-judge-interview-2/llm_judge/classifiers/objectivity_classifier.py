"""
This is the base class to implement different judges. This class takes in a MatchPair object and updates it with the judgment.
"""

import time
from adapters.adapter_factory import AdapterFactory
from adapters.types import (
    ConversationRole,
    Turn,
    Conversation,
    AdapterRateLimitException,
)
from llm_judge.database.models.questions import Question
from llm_judge.classifiers.classifier import Classifier


class ObjectivityClassifier(Classifier):
    def __init__(self, model: str = "gpt-4-turbo-preview", error_retry_limit=3):
        self.SYSTEM_PROMPT = """Decide whether the user question in the conversation have a objective, unique answer. Keep the following in mind when making this decision:
        1. Usually, multiple choice questions or fill in the blank questions have objective unique answers.
            - However, if the question is a multiple choice question with no options provided, then there's NO objective unique answer.
            - If only the options are provided but no question is provided in the conversation, then there's NO objective unique answer.
        2. If the question itself is unclear, then there's NO objective unique answer.
        3. If there are multiple possible answers for the same question, the question does NOT have an objective unique answer.
        4. We only care about whether the LAST user turn can have an objective answer. If the earlier parts of the conversation is discussing a question with an objective unique answer, but the last question is an open ended question, the conversation does NOT have a unique objective response.
        5. Questions that ask to define something, summarize something, write something, or explain something does not have an objective unique answer. 
        6. Questions that ask to translate something does have an objective unique answer.
        7. If the question require infomration in a plot or an image, the question is not objective.
        8. If the question has multiple sub-questions, then the question does not have an objective unique answer.

        Examples of questions with objective unique answers:
        1. User: How many hours are there in a year?
        2. User: Sort the following words in reverse alphabetical order: Google, Microsoft, Amazon, Facebook
        3. User: Select whether the following statement is always true, sometimes true, or never true.
            A proof should have more steps in the reason column than steps in the statement column.
            Never
            Sometimes
            Always
        
        Examples of questions with NO objective unique answer:
        1. User: Briefly explain symptoms of type I diabetes
        2. User: Is RX 7600 XT a good GPU?
        3. User: During the American Revolution, the Cherokee people had fought on the side of the British.  When they and the British were beaten, the Americans burned their crops and villages.  The Cherokee knew they could never win against Americans who were taking over their land so in 1795 they signed a treaty with the new United States government.  In the agreement, the Cherokee said they would stay on their lands in Georgia and other states peacefully.  In return, the US said that no white person could come into their lands without the permission of the Cherokee.
            Later during the War of 1812, the Cherokee actually helped the Americans fight against the British.  The Cherokee leader, Major Ridge, wanted his people to get along with Americans.  The problem was that in 1829, gold was found on Cherokee lands.  White settlers started to pour into the area which broke the treaty that the Cherokee had with the United States.  Andrew Jackson supported the settlers and asked Congress to remove the Cherokee.  The Cherokees were betrayed.
            In 1830, the Cherokee gave white settlers a warning and told them they had to move out.  They then burned down the houses of the white settlers.  White settlers throughout Georgia became angry and started attacking the Cherokee in return.
            Based on this reading, identify 3 cause and effect relationships:
            A - 
            B - 
            C -
        4. User: what is 10 * 10?\nAssistant: 100\nUser: ok

        Output your final decision by **strictly following the following format**:
        ```Answer: "[[OBJECTIVE]]"``` if the user question has an objective unique answer.
        ```Answer: "[[SUBJECTIVE]]"``` if the candidate does not have an objective unique answer.
        """
        self.CLASSIFY_PROMPT_TEMPLATE = """Conversation: {conversation} Decide whether the user question in the above conversation has an objective unique answer."""
        self.model = model
        self.error_retry_limit = error_retry_limit

    def generate_classification(self, question: Question):
        llm = AdapterFactory.get_adapter(self.model)
        conversation_string = question.conversation.convert_to_prompt()
        user_prompt = self.CLASSIFY_PROMPT_TEMPLATE.format(
            conversation=conversation_string
        )
        conv = Conversation(
            [
                Turn(role=ConversationRole.system, content=self.SYSTEM_PROMPT),
                Turn(role=ConversationRole.user, content=user_prompt),
            ]
        )
        input_conv = llm.convert_to_input(conv)

        num_retries = 0
        classification = "error"
        while classification == "error" and self.error_retry_limit > num_retries:
            try:
                judgment = llm.execute_sync(input_conv, temperature=0).response.content
            except AdapterRateLimitException as e:
                judgment = ""
                print("Rate limit exceeded, waiting 1 minute")
                time.sleep(60)

            if "[[OBJECTIVE]]" in judgment:
                classification = "objective"
            elif "[[SUBJECTIVE]]" in judgment:
                classification = "subjective"
            else:
                print(
                    f"** objecivity classifier - regenerate due to error. Judgement: {judgment}"
                )
            num_retries += 1

        return classification

    def get_class_names(self):
        return ["objective", "subjective"]
