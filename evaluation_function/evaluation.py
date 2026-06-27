import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from lf_toolkit.evaluation import Result, Params

load_dotenv()

BIOLOGY_TASK_PATH = Path(__file__).resolve().parent.parent / "biology.json"
DEFAULT_MODEL = "openai/gpt-4o-mini"
VALID_SCORES = {"Excellent", "Good", "Partial", "Misconception", "Incorrect"}

SYSTEM_PROMPT = """
You are an intelligent evaluation system. You will receive a prompt, a reference
answer, example student responses with score levels, and one new student
response.

Evaluate the new response based on the prompt, using similar guidelines to the
given example responses.

Choose exactly one score:
- Excellent
- Good
- Partial
- Misconception
- Incorrect

If the score is Excellent, give brief positive feedback. Otherwise, give concise
feedback in 1-2 sentences explaining how the student can improve.

Return only valid JSON in this exact shape:
{"Score": "Excellent", "Feedback": "Well done, that is correct!"}
""".strip()


def _load_biology_task() -> dict[str, Any]:
    with BIOLOGY_TASK_PATH.open(encoding="utf-8") as task_file:
        return json.load(task_file)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _build_user_prompt(task: dict[str, Any], response: Any, answer: Any) -> str:
    reference_answer = task.get("reference_answer") or answer
    evaluation_context = {
        "prompt": task.get("prompt"),
        "subject": task.get("subject"),
        "topic": task.get("topic"),
        "reference_answer": reference_answer,
        "example_responses": task.get("student_answers", []),
        "student_response_to_evaluate": response,
    }

    return (
        "Evaluate the student response using this context:\n"
        f"{_json_dumps(evaluation_context)}"
    )


def _parse_llm_result(content: str | None) -> tuple[str, str]:
    content = (content or "").strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return "Incorrect", content.strip() or "Unable to evaluate the response."
        try:
            data = json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return "Incorrect", content.strip() or "Unable to evaluate the response."

    score = str(data.get("Score") or data.get("score") or "").strip().title()
    feedback = str(data.get("Feedback") or data.get("feedback") or "").strip()

    if score not in VALID_SCORES:
        score = "Incorrect"

    if not feedback:
        feedback = "Unable to evaluate the response."

    return score, feedback


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

    task = _load_biology_task()
    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        result = Result(is_correct=False)
        result.add_feedback(
            "general",
            "Evaluation could not run because OPENROUTER_API_KEY is not configured.",
        )
        return result

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        max_retries=3,
    )

    llm_response = client.chat.completions.create(
        model=params.get("model", DEFAULT_MODEL),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(task, response, answer)},
        ],
        response_format={"type": "json_object"},
    )

    score, feedback = _parse_llm_result(llm_response.choices[0].message.content)
    result = Result(is_correct=score == "Excellent")

    result.add_feedback(
        "general",
        f"{score}: {feedback}",
    )

    return result
