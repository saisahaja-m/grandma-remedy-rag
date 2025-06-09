from ragas.metrics import faithfulness, answer_relevancy, ResponseGroundedness, ContextRelevance
from ragas import evaluate
from datasets import Dataset
from rag_func.constants.config import EVALUATION, ACTIVE_CONFIG
from trulens.providers.openai import OpenAI
from rag_func.constants.enums import EvaluatorTypesEnum, EvaluatingMetricsEnum
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric, AnswerRelevancyMetric, FaithfulnessMetric, ContextualRecallMetric
from typing import Dict
import requests
import os


def get_evaluator():
    eval_config = EVALUATION[ACTIVE_CONFIG["evaluation"]]
    eval_type = eval_config["type"]

    if eval_type == EvaluatorTypesEnum.RagasEvaluator.value:
        return RagasEvaluator(metrics=eval_config["metrics"])
    elif eval_type == EvaluatorTypesEnum.TrulensEvaluator.value:
        return TrulensEvaluator(model_name=eval_config['model_name'])
    elif eval_type == EvaluatorTypesEnum.DeepEvalEvaluator.value:
        return DeepEvalEvaluator(model_name=eval_config['model_name'])
    elif eval_type == EvaluatorTypesEnum.Custom.value:
        return CustomEvaluator()
    return None


class RagasEvaluator:
    def __init__(self, metrics=None):
        self.metric_mapping = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "groundedness": ResponseGroundedness(),
            "context_relevance": ContextRelevance()
        }

        if metrics is None:
            metrics = [EvaluatingMetricsEnum.Faithfulness.value,
                       EvaluatingMetricsEnum.AnswerRelevancy.value,
                       EvaluatingMetricsEnum.Groundedness.value,
                       EvaluatingMetricsEnum.ContextRelevance.value]

        self.metrics = [
            self.metric_mapping[metric] for metric in metrics if metric in self.metric_mapping]

    def evaluate(self, question, answer, contexts, ground_truths=None):
        if ground_truths is None:
            ground_truths = [""]

        data = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truths": [ground_truths]
        })

        result = evaluate(data, metrics=self.metrics)
        return result


class TrulensEvaluator:
    def __init__(self, model_name):
        self.provider = OpenAI(model_engine=model_name)

    def evaluate(self, question: str, answer: str, retrieved_context: list):
        scores = {}
        context_text = "\n".join(retrieved_context)

        groundedness_score, _ = self.provider.groundedness_measure_with_cot_reasons(answer, context_text)
        scores[EvaluatingMetricsEnum.Groundedness.value] = str(groundedness_score)

        answer_relevance_score, _ = self.provider.relevance_with_cot_reasons(question, answer)
        scores[EvaluatingMetricsEnum.AnswerRelevancy.value] = str(answer_relevance_score)

        context_relevance_score, _ = self.provider.context_relevance_with_cot_reasons(question,
                                                                                      context=context_text)
        scores[EvaluatingMetricsEnum.ContextRelevance.value] = str(context_relevance_score)

        correctness, _ = self.provider.correctness_with_cot_reasons(question, answer)
        scores[EvaluatingMetricsEnum.Correctness.value] = str(correctness)

        return scores

class DeepEvalEvaluator:
    def __init__(self, model_name):
        self.model_name = model_name

    def evaluate(self, question: str, answer: str, retrieved_context: list, expected_output: str):
        from deepeval import evaluate

        relevant_docs = [
            doc if isinstance(doc, str) else doc.page_content
            for doc in retrieved_context
        ]

        answer_relevance_metric = AnswerRelevancyMetric(
            threshold=0.0,
            model=self.model_name,
            include_reason=True
        )

        faithfulness_metric = FaithfulnessMetric(
            threshold=0.0,
            model=self.model_name,
            include_reason=True
        )

        context_relevancy_metric = ContextualRelevancyMetric(
            threshold=0.0,
            model=self.model_name,
            include_reason=True
        )

        context_recall_metric = ContextualRecallMetric(
            threshold=0.0,
            model=self.model_name,
            include_reason=True
        )
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=relevant_docs,
            expected_output=expected_output
        )
        results = evaluate(test_cases=[test_case],
                           metrics=[answer_relevance_metric, faithfulness_metric, context_relevancy_metric,
                                    context_recall_metric])

        return results

class CustomEvaluator:
    def __init__(self):
        pass

    def evaluate(self, user_query: str, context: str, llm_generated_answer: str,
                 actual_correct_answer: str) -> Dict[str, int]:
        scores = {}

        context_str = self.to_string(context)
        llm_answer_str = self.to_string(llm_generated_answer)

        scores["Faithfulness"] = int(self.evaluate_faithfulness(llm_answer_str, context_str))
        scores["Context Relevance"] = int(self.evaluate_context_relevance(user_query, context_str))
        scores["Answer Relevance"] = int(self.evaluate_answer_relevance(user_query, llm_answer_str))
        scores["Groundedness"] = int(self.evaluate_groundedness(llm_answer_str, context_str))
        scores["Context Recall"] = int(self.evaluate_context_recall(actual_correct_answer, context_str))

        return scores

    def evaluate_faithfulness(self, answer: str, context: str) -> int:
        prompt = (
            "You are an evaluator assessing the faithfulness of an answer based on the provided context. "
            "Faithfulness measures how much the answer accurately reflects the information in the context without introducing unsupported details. "
            "Please rate the faithfulness of the answer on a scale from 1 to 5, where:\n"
            "- 1: The answer contains mostly unsupported or incorrect information.\n"
            "- 5: The answer is fully consistent with the context and contains no unsupported details.\n\n"
            f"Context: {context}\n"
            f"Answer: {answer}\n"
            "Provide a single integer score between 1 and 5."
        )
        return self.get_llm_score(prompt)

    def evaluate_context_relevance(self, query: str, context: str) -> int:
        prompt = (
            "You are an evaluator assessing the relevance of a context to a user query. "
            "Context relevance measures how well the context provides information necessary to answer the query. "
            "Please rate the relevance of the context to the query on a scale from 1 to 5, where:\n"
            "- 1: The context is mostly irrelevant to the query.\n"
            "- 5: The context is highly relevant and directly addresses the query.\n\n"
            f"Query: {query}\n"
            f"Context: {context}\n"
            "Provide a single integer score between 1 and 5."
        )
        return self.get_llm_score(prompt)

    def evaluate_answer_relevance(self, query: str, answer: str) -> int:
        prompt = (
            "You are an evaluator assessing the relevance of an answer to a user query. "
            "Answer relevance measures how well the answer addresses the query's intent and provides pertinent information. "
            "Please rate the relevance of the answer to the query on a scale from 1 to 5, where:\n"
            "- 1: The answer is mostly irrelevant to the query.\n"
            "- 5: The answer fully addresses the query's intent and provides relevant information.\n\n"
            f"Query: {query}\n"
            f"Answer: {answer}\n"
            "Provide a single integer score between 1 and 5."
        )
        return self.get_llm_score(prompt)

    def evaluate_groundedness(self, answer: str, context: str) -> int:
        prompt = (
            "You are an evaluator assessing the groundedness of an answer based on the provided context. "
            "Groundedness measures the extent to which the answer's claims are supported by the context, focusing on factual accuracy and evidence. "
            "Please rate the groundedness of the answer on a scale from 1 to 5, where:\n"
            "- 1: The answer contains claims that are mostly unsupported by the context.\n"
            "- 5: All claims in the answer are fully supported by the context.\n\n"
            f"Context: {context}\n"
            f"Answer: {answer}\n"
            "Provide a single integer score between 1 and 5."
        )
        return self.get_llm_score(prompt)

    def evaluate_context_recall(self, ground_truth: str, context: str) -> int:
        prompt = (
            "You are an evaluator assessing context recall, which measures how well the context captures the information needed to produce the ground truth answer. "
            "Please rate the context's ability to provide all necessary information for the ground truth answer on a scale from 1 to 5, where:\n"
            "- 1: The context misses most information required for the ground truth answer.\n"
            "- 5: The context fully contains all information needed for the ground truth answer.\n\n"
            f"Ground Truth Answer: {ground_truth}\n"
            f"Context: {context}\n"
            "Provide a single integer score between 1 and 5."
        )
        return self.get_llm_score(prompt)

    def to_string(self, text) -> str:
        if isinstance(text, list):
            return " ".join(str(item) for item in text)
        return str(text)

    def get_llm_score(self, prompt: str) -> int:
        api_key = os.getenv("CLAUDE_API_KEY")
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        data = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 4096
        }

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data
        )

        if response.status_code != 200:
            raise Exception(f"Error from Claude API: {response.text}")

        return response.json()["content"][0]["text"]
