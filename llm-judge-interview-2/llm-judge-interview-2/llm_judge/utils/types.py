from pydantic import BaseModel
from typing import Optional, Dict, List
from enum import Enum


class LLMParamType(BaseModel):
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    prompt_modification: Optional[Dict[str, str]] = None


LLMParamsList = List[Dict[str, LLMParamType]]


def convert_to_llm_params_list(params: List[dict]) -> LLMParamsList:
    result = []
    for model_dict in params:
        for model_name, params in model_dict.items():
            model_params = LLMParamType(**params)
            result.append({model_name: model_params})
    return result


class PromptType(Enum):
    SYSTEM = "system"
    PROMPT_TEMPLATE = "prompt_template"
    REVERSED_PROMPT_TEMPLATE = "reversed_prompt_template"


class JudgeType(Enum):
    GENERATIVE_GROUND_TRUTH = "ground_truth"
    BASELINE_PAIRWISE = "baseline_pairwise"
    ALL_PAIRS = "all_pairs"
    EXACT_MATCH = "exact_match"
    DYNAMIC_FEW_SHOT = "dynamic_few_shot"
