from adapters.types import Conversation, Turn


def add_to_system_prompt(
    conversation: Conversation, additional_system_prompt: str
) -> Conversation:
    """
    Add an additional system prompt to the beginning of the conversation.
    """
    if conversation.turns[0].role == "system":
        original_prompt = conversation.turns[0].content
        conversation.turns[0] = Turn(
            role="system", content=f"{original_prompt}\n{additional_system_prompt}"
        )
    else:
        new_turn = Turn(role="system", content=additional_system_prompt)
        conversation.turns = [new_turn] + conversation.turns
    return conversation


def make_prompt_modification(
    conversation: Conversation, prompt_modification: dict[str, str]
) -> Conversation:
    for modification_alias, modification_prompt in prompt_modification.items():
        conversation = PROMPT_MODIFICATION_MAP[modification_alias](
            conversation, modification_prompt
        )
    return conversation


# Mapping of prompt modification names to functions that modify the conversation
# we don't want to modify this directory because it helps us keep track of existing Answers and how they were generated
# if a new modification is added, it should be added to this directory
PROMPT_MODIFICATION_MAP = {
    "cot": add_to_system_prompt,
}
