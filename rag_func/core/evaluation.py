from ragas.metrics import faithfulness, answer_relevancy, ResponseGroundedness, ContextRelevance
from ragas import evaluate
from datasets import Dataset
from rag_func.constants.config import EVALUATION, ACTIVE_CONFIG
from trulens.providers.openai import OpenAI
from rag_func.constants.enums import EvaluatorTypesEnum, EvaluatingMetricsEnum


def get_evaluator():
    eval_config = EVALUATION[ACTIVE_CONFIG["evaluation"]]
    eval_type = eval_config["type"]

    if eval_type == EvaluatorTypesEnum.RagasEvaluator.value:
        return RagasEvaluator(metrics=eval_config["metrics"])
    elif eval_type == EvaluatorTypesEnum.TrulensEvaluator.value:
        return TrulensEvaluator(model_name=eval_config['model_name'])
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