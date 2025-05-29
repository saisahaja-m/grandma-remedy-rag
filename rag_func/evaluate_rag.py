import json
import argparse
import pandas as pd
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import time
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
    # Retrieval
    start_time_retrieval = time.time()
    relevant_docs = rag_system["retriever"].invoke(user_input)
    end_time_retrieval = time.time()
    print(f"Time taken for retrieval: {end_time_retrieval - start_time_retrieval:.4f} seconds")
    docs = [doc for doc in relevant_docs if hasattr(doc, "page_content") and doc.page_content.strip()]

    if not docs:
        return "I'm sorry, I don't have specific information about that.", []

    # Reranking
    start_time_reranking = time.time()
    reranked_docs = rag_system["reranker"].rerank(user_input, docs)
    end_time_reranking = time.time()
    print(f"Time taken for reranking: {end_time_reranking - start_time_reranking:.4f} seconds")
    context = format_context_from_docs(reranked_docs)
    memories = []

    prompt = create_system_prompt(user_input, chat_history, context, memories)
    # Generation
    start_time_generation = time.time()
    response = rag_system["llm"].generate_response(prompt)
    end_time_generation = time.time()
    print(f"Time taken for generation: {end_time_generation - start_time_generation:.4f} seconds")

    return response, reranked_docs


def evaluate_response(rag_system, user_input, response, reranked_docs, ground_truth):
    context_docs = [doc.page_content for doc in reranked_docs]

    evaluation_result = rag_system["evaluator"].evaluate(
        user_input, response, context_docs, ground_truth
    )

    test_result = evaluation_result.test_results[0]
    metrics_data = test_result.metrics_data

    scores_dict = {}
    for metric in metrics_data:
        key = metric.name.lower().replace(' ', '_')
        scores_dict[key] = {
            "score": round(metric.score, 3),
            "reason": metric.reason
        }

    return scores_dict


def load_questions_and_ground_truths(filepath: str) -> List[Dict[str, str]]:
    try:
        df = pd.read_csv(filepath)
        return df[['question', 'answer']].to_dict('records')
    except Exception as e:
        print(f"Error loading questions and ground truths: {e}")
        return []


def run_evaluation(active_config: Dict, questions_file: str, output_file: str = None):
    rag_system = initialize_rag_system()

    qa_pairs = load_questions_and_ground_truths(questions_file)
    results = {
        "app_config": active_config,
        "metrics_summary": {
            "faithfulness": [],
            "answer_relevancy": [],
            "contextual_recall": [],
            "contextual_relevancy": []
        },
        "questions": []
    }

    for qa_pair in tqdm(qa_pairs):
        question = qa_pair["question"]
        ground_truth = qa_pair["answer"]

        answer, reranked_docs = process_query_with_rag(rag_system, question)

        if reranked_docs:
            # Evaluation
            start_time_evaluation = time.time()
            scores_dict = evaluate_response(
                rag_system=rag_system,
                user_input=question,
                response=answer,
                reranked_docs=reranked_docs,
                ground_truth=ground_truth
            )
            end_time_evaluation = time.time()
            print(f"Time taken for evaluation: {end_time_evaluation - start_time_evaluation:.4f} seconds")
        else:
            scores_dict = {
                "faithfulness": {"score": 0.0, "reason": "No relevant documents retrieved"},
                "answer_relevancy": {"score": 0.0, "reason": "No relevant documents retrieved"},
                "contextual_recall": {"score": 0.0, "reason": "No relevant documents retrieved"},
                "contextual_relevancy": {"score": 0.0, "reason": "No relevant documents retrieved"}
            }

        # Add question, answer, and scores to results
        results["questions"].append({
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": answer,
            "scores": scores_dict
        })

        # Collect scores for summary
        for metric in results["metrics_summary"]:
            results["metrics_summary"][metric].append(scores_dict[metric]["score"])

    # Calculate average scores for metrics summary
    for metric in results["metrics_summary"]:
        scores = results["metrics_summary"][metric]
        results["metrics_summary"][metric] = {
            "average_score": round(sum(scores) / len(scores), 3) if scores else 0.0,
            "num_evaluations": len(scores)
        }

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
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