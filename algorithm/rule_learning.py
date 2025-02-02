import os
import time
import json
import pickle
from typing import Callable, Dict, List, Optional
from algorithm.knowledge_graph import KnowledgeGraph
from algorithm.path_sampling import sample_bottom_rule
from algorithm.rule_generalization.GeneralizedRule_withConf import generalize_bottom_rule, GeneralizedRule

def AnyBURL(
    kg: KnowledgeGraph,
    sample_size: int,
    sat: float,
    ts: float,
    pc: float,
    max_total_time: float,
    alternate_cyclic_sampling: bool = True,
    quality_function: Optional[Callable[[GeneralizedRule], bool]] = None,
    # The authors say: "We have chosen the quality criteria Q to allow 
    # only those rules that generate at least two correct predictions, 
    # which is a very lax criteria."
    # I defined this default quality function below
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
    
    Returns:
      A dictionary of learned rules (keyed by their canonical string) mapping to GeneralizedRule objects.
    """
    # Setting the default quality function:
    if quality_function is None:
        def quality_fn(rule: GeneralizedRule) -> bool:
            return rule.head_groundings_count >= 2
        quality_function = quality_fn

    # Starting with path length = 2 (head triple + one body triple)
    n = 2
    # Initializing class attribute to store the rules:
    global_rules: Dict[str, GeneralizedRule] = {}
    
    iteration = 0
    total_start = time.time()
    while time.time() - total_start < max_total_time:
        iteration += 1

        # When n==3 and alternating cyclic sampling is enabled, choose mode based on iteration parity
        sample_mode = "cyclic" if (n == 3 and alternate_cyclic_sampling and iteration % 2 == 1) else "all"

        R_s: Dict[str, GeneralizedRule] = {}  # Rules discovered during this time span
        span_start = time.time()

        # Sampling bottom rules for the duration of the time span
        while time.time() - span_start < ts:
            bottom_rule = sample_bottom_rule(kg, n, direction_allowed="both") ### p ###
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
                    canonical_str = str(rule)  # for duplicate detection
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

    return global_rules
