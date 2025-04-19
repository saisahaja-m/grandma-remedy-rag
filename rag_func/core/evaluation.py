from ragas.metrics import faithfulness, answer_relevancy, ResponseGroundedness, ContextRelevance
from ragas import evaluate
from datasets import Dataset
from typing import List, Dict, Any, Union
from rag_func.config.config import EVALUATION, ACTIVE_CONFIG


def get_evaluator():
    eval_config = EVALUATION[ACTIVE_CONFIG["evaluation"]]
    eval_type = eval_config["type"]

    if eval_type == "ragas":
        return RagasEvaluator(metrics=eval_config["metrics"])



class RagasEvaluator():
    """RAGAS evaluator implementation"""

    def __init__(self, metrics=None):
        self.metric_mapping = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "groundedness": ResponseGroundedness(),
            "context_relevance": ContextRelevance()
        }

        if metrics is None:
            metrics = ["faithfulness", "answer_relevancy", "groundedness", "context_relevance"]

        self.metrics = [
            self.metric_mapping[metric] for metric in metrics if metric in self.metric_mapping
        ]

    def evaluate(self, question, answer, contexts, ground_truths=None):
        if ground_truths is None:
            ground_truths = [""]  # Default empty ground truth

        # Create a HuggingFace Dataset
        data = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truths": [ground_truths]
        })

        result = evaluate(data, metrics=self.metrics)
        return result