from ragas.metrics import faithfulness, answer_relevancy, ResponseGroundedness, ContextRelevance
from ragas import evaluate
from datasets import Dataset

def run_ragas_eval(question, answer, contexts, ground_truths=[""]):
    # Create a HuggingFace Dataset
    data = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truths": [ground_truths]
    })
    print(contexts)
    grounded_ness = ResponseGroundedness()
    context_relevance = ContextRelevance()

    result = evaluate(
        data,
        metrics=[
            faithfulness,
            answer_relevancy,
            grounded_ness,
            context_relevance
        ]
    )
    return result