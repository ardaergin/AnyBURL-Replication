import pickle
import time
from pathlib import Path
from typing import List, Tuple

from algorithm.knowledge_graph import KnowledgeGraph, Triple
from algorithm.rule_learning import AnyBURL
from algorithm.rule_prediction import RulePrediction

def load_triples(file_path: str) -> List[Tuple[str, str, str]]:
    """Load triples from a pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def evaluate_predictions(predictor: RulePrediction, test_triples: List[Triple], k: int = 10) -> dict:
    """
    Evaluate the predictor on test triples, computing Hits@k and MRR.
    """
    hits_at_k = 0
    reciprocal_ranks = []
    total = 0

    for test_triple in test_triples:
        # Predict tail entities
        predictions = predictor.predict_tail(
            subject=test_triple.subject,
            relation=test_triple.relation,
            k=k
        )
        
        # Check if correct tail entity is in top-k predictions
        pred_objects = [p[0] for p in predictions]
        if test_triple.object in pred_objects:
            hits_at_k += 1
            rank = pred_objects.index(test_triple.object) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
        
        total += 1

    return {
        f'hits@{k}': hits_at_k / total if total > 0 else 0,
        'mrr': sum(reciprocal_ranks) / total if total > 0 else 0
    }

def main():
    # Parameters
    TRAIN_PATH = "data/triples/WN18RR_triples_train.pkl"
    TEST_PATH = "data/triples/WN18RR_triples_test.pkl"
    LEARNING_TIME = 100  # seconds
    SAMPLE_SIZE = 500
    SAT_THRESHOLD = 0.99
    TIME_SPAN = 1.0
    PESSIMISTIC_CONSTANT = 5.0
    
    print("Loading data...")
    train_triples = load_triples(TRAIN_PATH)
    test_triples = load_triples(TEST_PATH)
    
    # Convert to Triple objects
    train_triples = [Triple.from_tuple(t) for t in train_triples]
    test_triples = [Triple.from_tuple(t) for t in test_triples]
    
    # Create knowledge graph from training data
    print("Creating knowledge graph...")
    kg = KnowledgeGraph(train_triples)
    
    print("Learning rules...")
    start_time = time.time()
    learned_rules = AnyBURL(
        kg=kg,
        sample_size=SAMPLE_SIZE,
        sat=SAT_THRESHOLD,
        ts=TIME_SPAN,
        pc=PESSIMISTIC_CONSTANT,
        max_total_time=LEARNING_TIME
    )
    learning_time = time.time() - start_time
    print(f"Rule learning completed in {learning_time:.2f} seconds")
    print(f"Total rules learned: {len(learned_rules)}")
    
    print("Creating predictor...")
    predictor = RulePrediction(learned_rules, kg)
    
    print("Evaluating on test set...")
    start_time = time.time()
    metrics = evaluate_predictions(predictor, test_triples, k=10)
    evaluation_time = time.time() - start_time
    
    print("\nResults:")
    print(f"Hits@10: {metrics['hits@10']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"Evaluation time: {evaluation_time:.2f} seconds")

if __name__ == "__main__":
    main()
