import os
from typing import Any
from dotenv import load_dotenv
from openai import OpenAI
from lf_toolkit.evaluation import Result, Params

load_dotenv()

def evaluation_function(
    response: Any,
    answer: Any,
    params: Params,
) -> Result:
    """
    Function used to evaluate a student response.
    ---
    The handler function passes three arguments to evaluation_function():

    - `response` which are the answers provided by the student.
    - `answer` which are the correct answers to compare against.
    - `params` which are any extra parameters that may be useful,
        e.g., error tolerances.

    The output of this function is what is returned as the API response
    and therefore must be JSON-encodable. It must also conform to the
    response schema.

    Any standard python library may be used, as well as any package
    available on pip (provided it is added to requirements.txt).

    The way you wish to structure you code (all in this function, or
    split into many) is entirely up to you. All that matters are the
    return types and that evaluation_function() is the main function used
    to output the evaluation response.
    """

    # result = Result(is_correct=response == answer)

    client = OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        max_retries=3,
    )

    SYSTEM_PROMPT = "You are a teaching assistant, give helpful feedback to the student."
    teacher_prompt = params.get('teacher_prompt', 'Evaluate the student response and provide helpful feedback.')

    prompt = SYSTEM_PROMPT + "\n" + teacher_prompt

    llm_response = client.chat.completions.create(
        model=params.get('model', 'openai/gpt-4o-mini'),
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": response},
        ],
    )

    result = Result(is_correct=True)

    result.add_feedback(
        "general",
        llm_response.choices[0].message.content,
    )

    return result