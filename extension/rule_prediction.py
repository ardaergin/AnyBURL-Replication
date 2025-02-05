from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import heapq
from .knowledge_graph import KnowledgeGraph, Triple
from .rule_generalization.GeneralizedRule_withConf import GeneralizedRule

class RulePrediction:
    def __init__(self, rules: Dict[str, GeneralizedRule], kg: KnowledgeGraph):
        """
        Initializing the prediction engine with learned rules and the knowledge graph.
        
        :param rules: Dictionary of learned rules (from AnyBURL)
        :param kg: Knowledge graph containing training data
        """
        self.rules = rules
        self.training_kg = kg
        
        # Indexing rules by head relation for faster lookup during prediction
        self.rules_by_relation = defaultdict(list)
        for rule in rules.values():
            # Grouping rules under the key of their head relation
            self.rules_by_relation[rule.generalized_head.relation].append(rule)
        
        # Sorting rules by confidence for each relation in descending order
        for relation in self.rules_by_relation:
            self.rules_by_relation[relation].sort(key=lambda x: x.confidence, reverse=True)
    
    def predict_tail(self, subject: str, relation: str, k: int = 10,
                     query_time: Optional[float] = None, tolerance: float = 0.0) -> List[Tuple[str, float]]:
        """
        Predicting top-k tail entities for a given (subject, relation) pair.
        Optionally, enforcing temporal consistency with a given query time.
        
        :param subject: Subject entity
        :param relation: Relation
        :param k: Number of predictions to return
        :param query_time: Optional timestamp (in the same unit as your KG) to filter predictions.
        :param tolerance: Tolerance for temporal matching.
        :return: List of (predicted_object, confidence) tuples, sorted by confidence.
        """
        # Creating a mapping from candidate objects to a list of confidence scores
        candidates = defaultdict(list)
        # Retrieving rules that are applicable for the specified relation
        applicable_rules = self.rules_by_relation.get(relation, [])
        
        # Iterating over each applicable rule to generate predictions
        for rule in applicable_rules:
            # Applying the rule to predict tail entities given the subject
            predictions = self._apply_rule_tail(rule, subject, query_time, tolerance)
            for obj, conf in predictions:
                # Collecting confidence scores for each candidate object
                candidates[obj].append(conf)
        
        # Aggregating confidences using the maximum strategy as described in the paper
        aggregated = []
        for obj, confidences in candidates.items():
            # Sorting confidences in descending order
            sorted_conf = sorted(confidences, reverse=True)
            # Creating a tuple of confidences padded with zeros for lexicographic comparison
            conf_tuple = tuple(sorted_conf + [0] * (k - len(sorted_conf)))
            aggregated.append((obj, conf_tuple))
        
        # Selecting the top-k predictions based on the aggregated confidence tuples
        top_k = heapq.nlargest(k, aggregated, key=lambda x: x[1])
        
        # Returning predictions with their highest confidence score
        return [(obj, conf_tuple[0]) for obj, conf_tuple in top_k]
    
    def predict_head(self, relation: str, object: str, k: int = 10,
                     query_time: Optional[float] = None, tolerance: float = 0.0) -> List[Tuple[str, float]]:
        """
        Predicting top-k head entities for a given (relation, object) pair.
        Optionally, enforcing temporal consistency with a given query time.
        
        :param relation: Relation
        :param object: Object entity
        :param k: Number of predictions to return
        :param query_time: Optional timestamp for temporal filtering.
        :param tolerance: Tolerance for temporal matching.
        :return: List of (predicted_subject, confidence) tuples, sorted by confidence.
        """
        # Creating a mapping from candidate subjects to a list of confidence scores
        candidates = defaultdict(list)
        # Retrieving rules that are applicable for the specified relation
        applicable_rules = self.rules_by_relation.get(relation, [])
        
        # Iterating over each applicable rule to generate head predictions
        for rule in applicable_rules:
            # Applying the rule to predict head entities given the object
            predictions = self._apply_rule_head(rule, object, query_time, tolerance)
            for subj, conf in predictions:
                # Collecting confidence scores for each candidate subject
                candidates[subj].append(conf)
        
        # Aggregating confidences using the maximum strategy
        aggregated = []
        for subj, confidences in candidates.items():
            # Sorting confidences in descending order
            sorted_conf = sorted(confidences, reverse=True)
            # Creating a tuple of confidences padded with zeros for lexicographic comparison
            conf_tuple = tuple(sorted_conf + [0] * (k - len(sorted_conf)))
            aggregated.append((subj, conf_tuple))
        
        # Selecting the top-k predictions based on the aggregated confidence tuples
        top_k = heapq.nlargest(k, aggregated, key=lambda x: x[1])
        # Returning predictions with their highest confidence score
        return [(subj, conf_tuple[0]) for subj, conf_tuple in top_k]
    
    def _apply_rule_tail(self, rule: GeneralizedRule, subject: str,
                          query_time: Optional[float], tolerance: float) -> List[Tuple[str, float]]:
        """
        Applying a rule to predict tail entities given a subject.
        If query_time is provided, only returning predictions that are temporally consistent.
        
        :param rule: The generalized rule to apply.
        :param subject: The subject entity provided in the query.
        :param query_time: Optional timestamp to enforce temporal consistency.
        :param tolerance: Tolerance for temporal matching.
        :return: List of (predicted_tail, confidence) tuples.
        """
        predictions = []
        grounding = {}
        
        # Binding the provided subject to the appropriate variable in the rule head
        if rule.generalized_head.subject == "Y":
            grounding["Y"] = subject
        elif rule.generalized_head.subject == "X":
            grounding["X"] = subject
        else:  # Handling a constant in the head subject position
            if rule.generalized_head.subject != subject:
                # Returning empty if the constant does not match the query subject
                return []
            grounding[subject] = subject
        
        # Attempting to complete the grounding using the rule body
        completed_groundings = self._complete_grounding(rule, grounding)
        
        # Extracting predictions from each completed grounding
        for grounding in completed_groundings:
            if rule.generalized_head.object == "X":
                if "X" in grounding:
                    candidate = grounding["X"]
                else:
                    continue
            elif rule.generalized_head.object == "Y":
                if "Y" in grounding:
                    candidate = grounding["Y"]
                else:
                    continue
            else:
                candidate = rule.generalized_head.object
            
            # If temporal filtering is requested, verifying the predicted fact's timestamp
            if query_time is not None:
                if not self.training_kg.has_fact_temporal(subject,
                                                          rule.generalized_head.relation,
                                                          candidate,
                                                          timestamp=query_time,
                                                          tolerance=tolerance):
                    continue
            # Appending the candidate and the rule's confidence to the predictions
            predictions.append((candidate, rule.confidence))
        return predictions
    
    def _apply_rule_head(self, rule: GeneralizedRule, object: str,
                          query_time: Optional[float], tolerance: float) -> List[Tuple[str, float]]:
        """
        Applying a rule to predict head entities given an object.
        If query_time is provided, only returning predictions that are temporally consistent.
        
        :param rule: The generalized rule to apply.
        :param object: The object entity provided in the query.
        :param query_time: Optional timestamp to enforce temporal consistency.
        :param tolerance: Tolerance for temporal matching.
        :return: List of (predicted_head, confidence) tuples.
        """
        predictions = []
        grounding = {}
        
        # Binding the provided object to the appropriate variable in the rule head
        if rule.generalized_head.object == "Y":
            grounding["Y"] = object
        elif rule.generalized_head.object == "X":
            grounding["X"] = object
        else:  # Handling a constant in the head object position
            if rule.generalized_head.object != object:
                # Returning empty if the constant does not match the query object
                return []
            grounding[object] = object
        
        # Attempting to complete the grounding using the rule body
        completed_groundings = self._complete_grounding(rule, grounding)
        
        # Iterating over each completed grounding to extract the head prediction
        for grounding in completed_groundings:
            if rule.generalized_head.subject == "X":
                if "X" in grounding:
                    candidate = grounding["X"]
                else:
                    continue
            elif rule.generalized_head.subject == "Y":
                if "Y" in grounding:
                    candidate = grounding["Y"]
                else:
                    continue
            else:
                candidate = rule.generalized_head.subject
            
            # If temporal filtering is requested, verifying the predicted fact's timestamp
            if query_time is not None:
                if not self.training_kg.has_fact_temporal(candidate,
                                                          rule.generalized_head.relation,
                                                          object,
                                                          timestamp=query_time,
                                                          tolerance=tolerance):
                    continue
            # Appending the candidate and the rule's confidence to the predictions
            predictions.append((candidate, rule.confidence))
        return predictions
    
    def _complete_grounding(self, rule: GeneralizedRule, partial_grounding: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Completing a partial grounding using the rule body.
        Iteratively binding variables by matching rule body triples with facts in the KG.
        
        :param rule: The generalized rule being applied.
        :param partial_grounding: A dictionary with some variables already bound.
        :return: A list of fully completed groundings (each a dict mapping variable names to entities).
                 Returning an empty list if no valid completions exist.
        """
        def _bind_variables(triple: Triple, current_grounding: Dict[str, str]) -> List[Dict[str, str]]:
            """
            Binding variables for a given triple in the rule body and a current grounding using the knowledge graph.
            
            :param triple: A triple (subject, relation, object) from the rule body.
            :param current_grounding: Current mapping of variables to entities.
            :return: A list of extended groundings with additional variable bindings.
            """
            new_groundings = []
            subj = str(triple.subject)
            obj = str(triple.object)
            
            # Handling the case when both subject and object are already bound
            if subj in current_grounding and obj in current_grounding:
                # Checking if the fact exists in the knowledge graph
                if self.training_kg.has_fact(current_grounding[subj], triple.relation, current_grounding[obj]):
                    new_groundings.append(current_grounding.copy())
            
            # Handling the case when the subject is bound
            elif subj in current_grounding:
                # Iterating over all possible objects linked to the bound subject
                for possible_obj in self.training_kg.adj[triple.relation].get(current_grounding[subj], []):
                    new_grounding = current_grounding.copy()
                    new_grounding[obj] = possible_obj
                    new_groundings.append(new_grounding)
                    
            # Handling the case when the object is bound
            elif obj in current_grounding:
                # Iterating over all possible subjects linked to the bound object
                for possible_subj in self.training_kg.adj_inv[triple.relation].get(current_grounding[obj], []):
                    new_grounding = current_grounding.copy()
                    new_grounding[subj] = possible_subj
                    new_groundings.append(new_grounding)

            return new_groundings

        # Initializing the list of current groundings with the provided partial grounding
        current_groundings = [partial_grounding]
        
        # Iterating over each triple in the rule body to extend the grounding
        for body_triple in rule.generalized_body:
            new_groundings = []
            for grounding in current_groundings:
                # Extending the current grounding using available facts
                new_groundings.extend(_bind_variables(body_triple, grounding))
            current_groundings = new_groundings
            if not current_groundings:
                # Stopping early if no valid groundings can be formed
                break
                
        return current_groundings
