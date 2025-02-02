from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import heapq
from .knowledge_graph import KnowledgeGraph, Triple
from .rule_generalization.GeneralizedRule_withConf import GeneralizedRule

class RulePrediction:
    def __init__(self, rules: Dict[str, GeneralizedRule], kg: KnowledgeGraph):
        """
        Initialize the prediction engine with learned rules and the knowledge graph.
        
        :param rules: Dictionary of learned rules (from AnyBURL)
        :param kg: Knowledge graph containing training data
        """
        self.rules = rules
        self.training_kg = kg
        
        # Index rules by head relation for faster lookup
        self.rules_by_relation = defaultdict(list)
        for rule in rules.values():
            self.rules_by_relation[rule.generalized_head.relation].append(rule)
        
        # Sort rules by confidence for each relation
        for relation in self.rules_by_relation:
            self.rules_by_relation[relation].sort(key=lambda x: x.confidence, reverse=True)

    def predict_tail(self, subject: str, relation: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Predict top-k tail entities for a given (subject, relation) pair.
        
        :param subject: Subject entity
        :param relation: Relation
        :param k: Number of predictions to return
        :return: List of (predicted_object, confidence) tuples, sorted by confidence
        """
        candidates = defaultdict(list)  # object -> list of confidences
        applicable_rules = self.rules_by_relation[relation]
        
        for rule in applicable_rules:
            predictions = self._apply_rule_tail(rule, subject)
            for obj, conf in predictions:
                candidates[obj].append(conf)
        
        # Aggregate confidences using max strategy as described in the paper
        aggregated = []
        for obj, confidences in candidates.items():
            # Sort confidences in descending order
            sorted_conf = sorted(confidences, reverse=True)
            # Create tuple of confidences for lexicographic comparison
            conf_tuple = tuple(sorted_conf + [0] * (k - len(sorted_conf)))
            aggregated.append((obj, conf_tuple))
        
        # Get top-k predictions based on confidence tuples
        top_k = heapq.nlargest(k, aggregated, key=lambda x: x[1])
        
        # Return predictions with their highest confidence score
        return [(obj, conf_tuple[0]) for obj, conf_tuple in top_k]

    def predict_head(self, relation: str, object: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Predict top-k head entities for a given (relation, object) pair.
        
        :param relation: Relation
        :param object: Object entity
        :param k: Number of predictions to return
        :return: List of (predicted_subject, confidence) tuples, sorted by confidence
        """
        candidates = defaultdict(list)  # subject -> list of confidences
        applicable_rules = self.rules_by_relation[relation]
        
        for rule in applicable_rules:
            predictions = self._apply_rule_head(rule, object)
            for subj, conf in predictions:
                candidates[subj].append(conf)
        
        # Aggregate confidences using max strategy
        aggregated = []
        for subj, confidences in candidates.items():
            sorted_conf = sorted(confidences, reverse=True)
            conf_tuple = tuple(sorted_conf + [0] * (k - len(sorted_conf)))
            aggregated.append((subj, conf_tuple))
        
        top_k = heapq.nlargest(k, aggregated, key=lambda x: x[1])
        return [(subj, conf_tuple[0]) for subj, conf_tuple in top_k]

    def _apply_rule_tail(self, rule: GeneralizedRule, subject: str) -> List[Tuple[str, float]]:
        """
        Apply a rule to predict tail entities for a given subject.
        """
        predictions = []
        grounding = {}
        
        # Set the subject in the grounding
        if rule.generalized_head.subject == "Y":
            grounding["Y"] = subject
        elif rule.generalized_head.subject == "X":
            grounding["X"] = subject
        else:  # Constant in head subject position
            if rule.generalized_head.subject != subject:
                return []
            grounding[subject] = subject
            
        # Try to complete the grounding using the body
        completed_groundings = self._complete_grounding(rule, grounding)
        
        # Extract predictions from completed groundings
        for grounding in completed_groundings:
            if rule.generalized_head.object == "X":
                if "X" in grounding:
                    predictions.append((grounding["X"], rule.confidence))
            elif rule.generalized_head.object == "Y":
                if "Y" in grounding:
                    predictions.append((grounding["Y"], rule.confidence))
            else:  # Constant in head object position
                predictions.append((rule.generalized_head.object, rule.confidence))
                
        return predictions

    def _apply_rule_head(self, rule: GeneralizedRule, object: str) -> List[Tuple[str, float]]:
        """
        Apply a rule to predict head entities for a given object.
        """
        predictions = []
        grounding = {}
        
        # Set the object in the grounding
        if rule.generalized_head.object == "Y":
            grounding["Y"] = object
        elif rule.generalized_head.object == "X":
            grounding["X"] = object
        else:  # Constant in head object position
            if rule.generalized_head.object != object:
                return []
            grounding[object] = object
            
        # Try to complete the grounding using the body
        completed_groundings = self._complete_grounding(rule, grounding)
        
        # Extract predictions from completed groundings
        for grounding in completed_groundings:
            if rule.generalized_head.subject == "X":
                if "X" in grounding:
                    predictions.append((grounding["X"], rule.confidence))
            elif rule.generalized_head.subject == "Y":
                if "Y" in grounding:
                    predictions.append((grounding["Y"], rule.confidence))
            else:  # Constant in head subject position
                predictions.append((rule.generalized_head.subject, rule.confidence))
                
        return predictions

    def _complete_grounding(self, rule: GeneralizedRule, partial_grounding: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Try to complete a partial grounding using the rule body.
        Returns a list of completed groundings (can be empty if no valid completions exist).
        """
        def _bind_variables(triple: Triple, current_grounding: Dict[str, str]) -> List[Dict[str, str]]:
            new_groundings = []
            subj = str(triple.subject)
            obj = str(triple.object)
            
            # Both bound
            if subj in current_grounding and obj in current_grounding:
                if self.training_kg.has_fact(current_grounding[subj], triple.relation, current_grounding[obj]):
                    new_groundings.append(current_grounding.copy())
                    
            # Subject bound
            elif subj in current_grounding:
                for possible_obj in self.training_kg.adj[triple.relation].get(current_grounding[subj], []):
                    new_grounding = current_grounding.copy()
                    new_grounding[obj] = possible_obj
                    new_groundings.append(new_grounding)
                    
            # Object bound
            elif obj in current_grounding:
                for possible_subj in self.training_kg.adj_inv[triple.relation].get(current_grounding[obj], []):
                    new_grounding = current_grounding.copy()
                    new_grounding[subj] = possible_subj
                    new_groundings.append(new_grounding)
                    
            return new_groundings

        current_groundings = [partial_grounding]
        
        # Process each body triple
        for body_triple in rule.generalized_body:
            new_groundings = []
            for grounding in current_groundings:
                new_groundings.extend(_bind_variables(body_triple, grounding))
            current_groundings = new_groundings
            if not current_groundings:  # Early stopping if no valid groundings
                break
                
        return current_groundings
