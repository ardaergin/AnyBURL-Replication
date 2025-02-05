import os
import time
import json
import pickle
from typing import Callable, Dict, List, Optional
from .knowledge_graph import KnowledgeGraph
from .path_sampling import sample_bottom_rule
from .rule_generalization.GeneralizedRule_withConf import generalize_bottom_rule, GeneralizedRule

def AnyBURL(
    kg: KnowledgeGraph,
    sample_size: int,
    sat: float,
    ts: float,
    pc: float,
    max_total_time: float,
    alternate_cyclic_sampling: bool = True,
    quality_function: Optional[Callable[[GeneralizedRule], bool]] = None,
    dataset_name: Optional[str] = None,
    temporal_window: Optional[float] = None  # NEW: Maximum allowed gap (in seconds) between consecutive events.
) -> Dict[str, GeneralizedRule]:
    """
    Anytime Bottom-up Rule Learning implementation that follows "Algorithm 1" in Meilicke et al. (2019).
    
    Parameters:
      - kg: The knowledge graph.
      - sample_size: Number of body groundings to sample when calculating rule confidence.
      - sat: Saturation threshold. If in a time span the fraction of already-seen rules is greater than sat,
             increase the path length.
      - quality_function: A callable Q(r) that takes a rule (GeneralizedRule) and returns True if the rule is
                          of sufficient quality.
      - ts: The duration (in seconds) of one learning “time span”.
      - pc: The pessimistic constant used for Laplace smoothing in the confidence calculation.
      - max_total_time: Total time (in seconds) to run the learning process.
      - alternate_cyclic_sampling: When True and n==3, alternate between sampling only cyclic paths and all paths.
      - dataset_name: Optionally, the name of the dataset (used for logging).
      - temporal_window: Optional maximum gap (in seconds) allowed between consecutive timestamps in a sampled path.
    
    Returns:
      A dictionary of learned rules (keyed by their canonical string) mapping to GeneralizedRule objects.
    """
    # For saving the rules
    os.makedirs("rules", exist_ok=True)
    if dataset_name is None:
        session_timestamp = time.strftime('%Y%m%d_%H%M%S')
        session_filename = f"rules_session_{session_timestamp}.txt"
    else:
        session_filename = f"rules_session_{dataset_name}.txt"
    session_filepath = os.path.join("rules", session_filename)
    
    with open(session_filepath, "a", encoding="utf-8") as fout:
        if dataset_name is None:
            fout.write(f"# New training session started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        else:
            fout.write(f"# New training session for dataset {dataset_name} started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Setting the default quality function:
    if quality_function is None:
        def quality_fn(rule: GeneralizedRule) -> bool:
            return rule.head_groundings_count >= 2
        quality_function = quality_fn

    # Starting with path length = 2 (head triple + one body triple)
    n = 2
    # Dictionary to store all learned rules (for duplicate filtering)
    global_rules: Dict[str, GeneralizedRule] = {}
    
    iteration = 0
    total_start = time.time()
    while time.time() - total_start < max_total_time:
        iteration += 1

        # Alternating between cyclic and all sampling
        # Authors mention that it is difficult to find cyclics after n==3
        # and they say they turn this off after n==3
        if n == 3 and alternate_cyclic_sampling and iteration % 2 == 1:
            sample_mode = "cyclic"
        else:
            sample_mode = "all"

        R_s: Dict[str, GeneralizedRule] = {}  # Rules discovered in the current time span
        span_start = time.time()

        # Sample bottom rules for the duration of the time span
        while time.time() - span_start < ts:
            # Pass along the temporal_window so that the sampling process enforces time constraints
            bottom_rule = sample_bottom_rule(kg, n, direction_allowed="both", temporal_window=temporal_window)
            if bottom_rule is None:
                continue

            # If in cyclic-only mode, we skip the acyclic bottom rules we find
            if sample_mode == "cyclic" and not bottom_rule.is_cyclical:
                continue

            # Generating generalized rules from the bottom rule we sampled
            generalized_rules = generalize_bottom_rule(bottom_rule) ### R_p ###

            # Calculating the confidence of each of these generalized rules
            for rule in generalized_rules:
                rule.calculate_confidence(kg, sample_size=sample_size, pc=pc)
                if quality_function(rule):
                    canonical_str = rule.to_logical_string()  # for duplicate detection
                    R_s[canonical_str] = rule

        # Checking saturation (the fraction of new rules that were already seen)
        saturation = 0.0
        if R_s:
            common = set(R_s.keys()).intersection(set(global_rules.keys()))
            saturation = len(common) / len(R_s)
            # Increase path length if saturation is above the threshold
            if saturation > sat:
                n += 1  

        global_rules.update(R_s)
        print(f"Iteration {iteration}: n = {n}, new rules = {len(R_s)}, saturation = {saturation:.2f}, "
              f"total rules learned = {len(global_rules)}")
        
        with open(session_filepath, "a", encoding="utf-8") as fout:
            fout.write(f"\n# Iteration {iteration}: n = {n}, new rules = {len(R_s)}, "
                       f"saturation = {saturation:.4f}, total rules learned = {len(global_rules)}\n")
            for rule_str in R_s.keys():
                fout.write(rule_str + "\n")

    return global_rules
