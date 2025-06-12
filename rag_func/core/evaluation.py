from abc import ABC, abstractmethod
from ragas.metrics import faithfulness, answer_relevancy, ResponseGroundedness, ContextRelevance
from ragas import evaluate
from datasets import Dataset
from rag_func.constants.config import EVALUATION, ACTIVE_CONFIG
from trulens.providers.openai import OpenAI
from rag_func.constants.enums import EvaluatorTypesEnum, EvaluatingMetricsEnum
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric, AnswerRelevancyMetric, FaithfulnessMetric, ContextualRecallMetric
from typing import Dict, List, Union, Optional

class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, question: str, answer: str, contexts: List[str], ground_truth: Optional[str] = None) -> Union[Dict, List]:
        pass

class RagasEvaluator(BaseEvaluator):
    def __init__(self, metrics: Optional[List[str]] = None):
        self.metric_mapping = {
            EvaluatingMetricsEnum.Faithfulness.value: faithfulness,
            EvaluatingMetricsEnum.AnswerRelevancy.value: answer_relevancy,
            EvaluatingMetricsEnum.Groundedness.value: ResponseGroundedness(),
            EvaluatingMetricsEnum.ContextRelevance.value: ContextRelevance()
        }

        if metrics is None:
            metrics = [
                EvaluatingMetricsEnum.Faithfulness.value,
                EvaluatingMetricsEnum.AnswerRelevancy.value,
                EvaluatingMetricsEnum.Groundedness.value,
                EvaluatingMetricsEnum.ContextRelevance.value
            ]

        self.metrics = [self.metric_mapping[metric] for metric in metrics if metric in self.metric_mapping]

    def evaluate(self, question: str, answer: str, contexts: List[str], ground_truth: Optional[str] = None) -> Dict:
        ground_truths = [ground_truth] if ground_truth else [""]
        data = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truths": [ground_truths]
        })
        result = evaluate(data, metrics=self.metrics)
        return result

class TrulensEvaluator(BaseEvaluator):
    def __init__(self, model_name: str):
        self.provider = OpenAI(model_engine=model_name)

    def evaluate(self, question: str, answer: str, contexts: List[str], ground_truth: Optional[str] = None) -> Dict:
        scores = {}
        context_text = "\n".join(contexts)

        groundedness_score, _ = self.provider.groundedness_measure_with_cot_reasons(answer, context_text)
        scores[EvaluatingMetricsEnum.Groundedness.value] = str(groundedness_score)

        answer_relevance_score, _ = self.provider.relevance_with_cot_reasons(question, answer)
        scores[EvaluatingMetricsEnum.AnswerRelevancy.value] = str(answer_relevance_score)

        context_relevance_score, _ = self.provider.context_relevance_with_cot_reasons(question, context_text)
        scores[EvaluatingMetricsEnum.ContextRelevance.value] = str(context_relevance_score)

        correctness, _ = self.provider.correctness_with_cot_reasons(question, answer)
        scores[EvaluatingMetricsEnum.Correctness.value] = str(correctness)

        return scores

class DeepEvalEvaluator(BaseEvaluator):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def evaluate(self, question: str, answer: str, contexts: List[str], ground_truth: Optional[str] = None):
        from deepeval import evaluate

        relevant_docs = [doc if isinstance(doc, str) else doc.page_content for doc in contexts]
        answer_relevance_metric = AnswerRelevancyMetric(threshold=0.0, model=self.model_name, include_reason=True)
        faithfulness_metric = FaithfulnessMetric(threshold=0.0, model=self.model_name, include_reason=True)
        context_relevancy_metric = ContextualRelevancyMetric(threshold=0.0, model=self.model_name, include_reason=True)
        context_recall_metric = ContextualRecallMetric(threshold=0.0, model=self.model_name, include_reason=True)

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=relevant_docs,
            expected_output=ground_truth or ""
        )
        results = evaluate(
            test_cases=[test_case],
            metrics=[answer_relevance_metric, faithfulness_metric, context_relevancy_metric, context_recall_metric]
        )
        return results


class EvaluatorFactory:
    @staticmethod
    def get_evaluator() -> BaseEvaluator:
        eval_config = EVALUATION[ACTIVE_CONFIG["evaluation"]]
        eval_type = eval_config["type"]

        evaluator_classes = {
            EvaluatorTypesEnum.RagasEvaluator.value: lambda: RagasEvaluator(metrics=eval_config.get("metrics")),
            EvaluatorTypesEnum.TrulensEvaluator.value: lambda: TrulensEvaluator(model_name=eval_config["model_name"]),
            EvaluatorTypesEnum.DeepEvalEvaluator.value: lambda: DeepEvalEvaluator(model_name=eval_config["model_name"])
        }

        evaluator_class = evaluator_classes.get(eval_type)
        if evaluator_class is None:
            raise ValueError(f"Unsupported evaluator type: {eval_type}")

        return evaluator_class()

def get_evaluator() -> BaseEvaluator:
    return EvaluatorFactory.get_evaluator()
