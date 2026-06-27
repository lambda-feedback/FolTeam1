import os
import unittest
from unittest.mock import Mock, patch

from .evaluation import Params, evaluation_function

class TestEvaluationFunction(unittest.TestCase):
    """
    TestCase Class used to test the algorithm.
    ---
    Tests are used here to check that the algorithm written
    is working as it should.

    It's best practise to write these tests first to get a
    kind of 'specification' for how your algorithm should
    work, and you should run these tests before committing
    your code to AWS.

    Read the docs on how to use unittest here:
    https://docs.python.org/3/library/unittest.html

    Use evaluation_function() to check your algorithm works
    as it should.
    """

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("evaluation_function.evaluation.OpenAI")
    def test_evaluation(self, openai_class):
        llm_response = Mock()
        llm_response.choices = [
            Mock(
                message=Mock(
                    content='{"Score": "Excellent", "Feedback": "Well done."}'
                )
            )
        ]
        openai_class.return_value.chat.completions.create.return_value = llm_response

        response, answer, params = "Hello, World", "Hello, World", Params()

        result = evaluation_function(response, answer, params).to_dict()

        self.assertEqual(result.get("is_correct"), True)
        self.assertTrue(result.get("feedback"))
        self.assertIn(
            "Explain how photosynthesis",
            openai_class.return_value.chat.completions.create.call_args.kwargs[
                "messages"
            ][1]["content"],
        )
