import os
import pickle
import time
from pathlib import Path
from typing import List, Tuple

from replication import Triple, KnowledgeGraph, AnyBURL, RulePrediction

def load_triples(file_path: str) -> List[Tuple[str, str, str]]:
    """Loading triples from a pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def evaluate_predictions(predictor: RulePrediction, kg: KnowledgeGraph, test_triples: List[Triple], k: int = 10) -> dict:
    """
    Evaluating the predictor on test triples, computing Hits@k, Hits@1, and MRR.
    """
    hits_at_k = 0
    hits_at_1 = 0
    reciprocal_ranks = []
    total = 0

    for test_triple in test_triples:
        subject = test_triple.subject
        relation = test_triple.relation
        true_object = test_triple.object

        # (1) Getting raw predictions
        predictions = predictor.predict_tail(subject, relation, k=k)
        
        # (2) Filtering out known objects (except for the test triple's true object)
        known_objs = kg.adj.get(relation, {}).get(subject, set())
        filtered_candidates = []
        for (obj, conf) in predictions:
            # Keeping if not already known (or if it is the test triple's object)
            if (obj == true_object) or (obj not in known_objs):
                filtered_candidates.append((obj, conf))
        
        # (3) Ranking candidates by confidence in descending order
        filtered_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # (4) Extracting candidate objects and computing Hits and reciprocal rank
        pred_objects = [p[0] for p in filtered_candidates]
        if len(pred_objects) > 0 and pred_objects[0] == true_object:
            hits_at_1 += 1
        if true_object in pred_objects[:k]:
            hits_at_k += 1
        
        if true_object in pred_objects:
            rank = pred_objects.index(true_object) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
        
        total += 1

    return {
        'hits@1': hits_at_1 / total if total > 0 else 0,
        'hits@10': hits_at_k / total if total > 0 else 0,
        'mrr': sum(reciprocal_ranks) / total if total > 0 else 0
    }

def run_experiment(train_path: str, test_path: str, dataset_name: str, learning_time: float,
                   sample_size: int, sat_threshold: float, time_span: float, pessimistic_constant: float):
    print(f"\n==================== Dataset: {dataset_name} | Learning Time: {learning_time} sec ====================")
    
    # Load data
    train_triples = load_triples(train_path)
    test_triples = load_triples(test_path)
    
    # Raw tuples into Triple objects
    train_triples = [Triple.from_tuple(t) for t in train_triples]
    test_triples = [Triple.from_tuple(t) for t in test_triples]
    
    # KG
    print("Creating knowledge graph...")
    kg = KnowledgeGraph(train_triples)
    
    # Learning rules using AnyBURL
    print("Learning rules...")
    start_time = time.time()
    learned_rules = AnyBURL(
        kg=kg,
        sample_size=sample_size,
        sat=sat_threshold,
        ts=time_span,
        pc=pessimistic_constant,
        max_total_time=learning_time
    )
    elapsed_learning = time.time() - start_time
    print(f"Rule learning completed in {elapsed_learning:.2f} seconds")
    print(f"Total rules learned: {len(learned_rules)}")
    
    # Creating the predictor using the learned rules
    print("Creating predictor...")
    predictor = RulePrediction(learned_rules, kg)
    
    # Evaluating on the test set
    print("Evaluating on test set...")
    start_eval = time.time()
    metrics = evaluate_predictions(predictor, kg, test_triples, k=10)
    elapsed_eval = time.time() - start_eval
    
    print("\nResults:")
    print(f"Hits@1: {metrics['hits@1']:.4f}")
    print(f"Hits@10: {metrics['hits@10']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"Evaluation time: {elapsed_eval:.2f} seconds")
    
    return metrics

def replication():
    datasets = [
        ("data/triples/FB15k237_triples_train.pkl", "data/triples/FB15k237_triples_test.pkl", "FB15k237"),
        ("data/triples/WN18RR_triples_train.pkl", "data/triples/WN18RR_triples_test.pkl", "WN18RR"),
        ("data/triples/YAGO_triples_train.pkl", "data/triples/YAGO_triples_test.pkl", "YAGO")
    ]
    learning_times = [10, 100, 1000] # took out 10000
    
    # Parameters
    SAMPLE_SIZE = 100
    SAT_THRESHOLD = 0.20
    TIME_SPAN = 1.0
    PESSIMISTIC_CONSTANT = 5.0
    
    # Running experiments at different learning time limits for each dataset
    results = {}
    for train_path, test_path, ds_name in datasets:
        results[ds_name] = {}
        for lt in learning_times:
            metrics = run_experiment(
                train_path=train_path,
                test_path=test_path,
                learning_time=lt,
                sample_size=SAMPLE_SIZE,
                sat_threshold=SAT_THRESHOLD,
                time_span=TIME_SPAN,
                pessimistic_constant=PESSIMISTIC_CONSTANT,
                dataset_name=ds_name,
            )
            results[ds_name][lt] = metrics
            
    # Saving the results to a file
    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # Printing a summary of the results
    print("\n==================== Summary ====================")
    for ds_name, res in results.items():
        print(f"\nDataset: {ds_name}")
        for lt, metrics in res.items():
            print(f"Learning Time: {lt} sec -> Hits@1: {metrics['hits@1']:.4f}, Hits@10: {metrics['hits@10']:.4f}, MRR: {metrics['mrr']:.4f}")

if __name__ == "__main__":
    replication()
