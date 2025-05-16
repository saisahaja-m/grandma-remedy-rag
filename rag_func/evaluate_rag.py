import argparse
import json
import pandas as pd
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from rag_func.core.data_processing import load_and_process_documents
from rag_func.core.retrieval import get_retriever
from rag_func.core.generation import get_llm_model
from rag_func.core.evaluation import get_evaluator
from rag_func.core.reranking import get_reranker
from rag_func.utils.helpers import format_context_from_docs, create_system_prompt
from rag_func.constants.config import ACTIVE_CONFIG

load_dotenv()


def initialize_rag_system():
    docs = load_and_process_documents()
    retriever = get_retriever(docs)
    reranker = get_reranker()
    llm = get_llm_model()
    evaluator = get_evaluator()

    return {
        "docs": docs,
        "retriever": retriever,
        "reranker": reranker,
        "llm": llm,
        "evaluator": evaluator
    }


def process_query_with_rag(rag_system, user_input, chat_history="") -> Tuple[str, List]:
    relevant_docs = rag_system["retriever"].invoke(user_input)
    docs = [doc for doc in relevant_docs if hasattr(doc, "page_content") and doc.page_content.strip()]

    if not docs:
        return "I'm sorry, I don't have specific information about that.", []

    reranked_docs = rag_system["reranker"].rerank(user_input, docs)
    context = format_context_from_docs(reranked_docs)
    memories = []

    prompt = create_system_prompt(user_input, chat_history, context, memories)
    response = rag_system["llm"].generate_response(prompt)

    return response, reranked_docs


def evaluate_response(rag_system, user_input, response, reranked_docs):
    context_docs = [doc.page_content for doc in reranked_docs]

    evaluation_result = rag_system["evaluator"].evaluate(
        user_input, response, context_docs
    )

    return evaluation_result


def load_questions_and_ground_truths(filepath: str) -> List[Dict[str, str]]:
    try:
        df = pd.read_csv(filepath)
        return df[['question', 'answer']].to_dict('records')
    except Exception as e:
        print(f"Error loading questions and ground truths: {e}")
        return []


def run_evaluation(active_config: Dict, questions_file: str, output_file: str = None):
    rag_system = initialize_rag_system()
    eval_result = []

    qa_pairs = load_questions_and_ground_truths(questions_file)
    results = {
        "app_config": active_config,
        "metrics": {
            "faithfulness": [],
            "answer_relevancy": [],
            "response_groundedness": [],
            "context_relevance": []
        },
        "questions": []
    }

    for qa_pair in tqdm(qa_pairs):
        question = qa_pair["question"]
        ground_truth = qa_pair["answer"]

        answer, reranked_docs = process_query_with_rag(rag_system, question)

        if reranked_docs:
            eval_result = evaluate_response(
                rag_system=rag_system,
                user_input=question,
                response=answer,
                reranked_docs=reranked_docs
            )

            test_result = eval_result.test_results[0]
            metrics_data = test_result.metrics_data

            scores_dict = {}
            for metric in metrics_data:
                key = metric.name.lower().replace(' ', '_')
                scores_dict[key] = metric.score

            # Add scores to metrics
            results["metrics"]["faithfulness"].append(scores_dict.get("faithfulness", 0.0))
            results["metrics"]["answer_relevancy"].append(scores_dict.get("answer_relevancy", 0.0))
            results["metrics"]["response_groundedness"].append(scores_dict.get("nv_response_groundedness", 0.0))
            results["metrics"]["context_relevance"].append(scores_dict.get("nv_context_relevance", 0.0))
        else:
            # No relevant documents found
            scores_dict = {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "nv_response_groundedness": 0.0,
                "nv_context_relevance": 0.0
            }

        # Add question, answer, and scores to results
        results["questions"].append({
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": answer,
            "scores": {
                "faithfulness": scores_dict.get("faithfulness", 0.0),
                "answer_relevancy": scores_dict.get("answer_relevancy", 0.0),
                "response_groundedness": scores_dict.get("nv_response_groundedness", 0.0),
                "context_relevance": scores_dict.get("nv_context_relevance", 0.0)
            }
        })

    # Calculate average scores
    for metric in results["metrics"]:
        if results["metrics"][metric]:
            results["metrics"][metric] = round(sum(results["metrics"][metric]) / len(results["metrics"][metric]), 3)
        else:
            results["metrics"][metric] = 0.0

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        with open(output_file, 'w') as f:
            json.dump(eval_result, f, indent=2)
        print(f"Results saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline on a set of questions")
    parser.add_argument("--questions", type=str, required=True,
                        help="Path to CSV file with questions and ground truths")
    parser.add_argument("--output", type=str, help="Path to save evaluation results (JSON)")

    args = parser.parse_args()

    results = run_evaluation(ACTIVE_CONFIG, args.questions, args.output)


if __name__ == "__main__":
    main()