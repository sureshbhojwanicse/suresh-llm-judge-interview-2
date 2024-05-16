from llm_judge.database.models.prompts import Prompt
from llm_judge.utils.types import JudgeType, PromptType

"""
Edit the following parameters and execute a file to create a new Prompt object in Mongodb.
"""

description = (
    "This is a baseline pairwise comparison judge reverse prompt template for DeepAI."
)
tags = ["deepai", JudgeType.BASELINE_PAIRWISE, PromptType.REVERSED_PROMPT_TEMPLATE]
content = """[Conversation]\n{question}\n\n[The Start of Candidate Answer]\n{candidate}\n[The End of Candidate Answer]\n\n[The Start of Reference Answer]\n{reference}\n[The End of Reference Answer]\n\nDecide if the candidate answer is as good as the reference answer."""
args = {}

if __name__ == "__main__":
    prompt = Prompt(description=description, tags=tags, content=content, args=args)
    prompt = prompt.get_or_save()
    print(f"Prompt saved with id: {prompt.id}")
