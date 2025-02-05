from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Any, Optional
import random
from ..knowledge_graph import Triple, KnowledgeGraph
from ..path_sampling import BottomRule


@dataclass
class GeneralizedRule:
    bottom_rule: BottomRule
    rule_type: str  # "AC1", "AC2", or "C"
    AC1_rule_variant: Optional[str] = None  # can be 'Y_as_constant' or 'X_as_constant'
    
    # Mappings:
    node_mappings: Dict[str, str] = field(init=False, default_factory=dict)

    # We also store the final generalized head and body:
    generalized_head: Triple = field(init=False)
    generalized_body: List[Triple] = field(init=False, default_factory=list)
    
    # (Approximate) Confidence-related
    confidence: float = field(init=False, default=0.0)
    body_groundings_count: int = field(init=False, default=0)
    head_groundings_count: int = field(init=False, default=0)

    def __post_init__(self):
        if self.bottom_rule is None:
            self.node_mappings = {}
            self.generalized_head = None
            self.generalized_body = []
            self.confidence = None
            return

        # Validate rule type and AC1 variant
        if self.rule_type not in ["AC1", "AC2", "C"]:
            raise ValueError("rule_type must be one of: AC1, AC2, C")
        if self.rule_type == "AC1" and self.bottom_rule.is_cyclical and self.AC1_rule_variant not in ["Y_as_constant", "X_as_constant"]:
            raise ValueError("AC1 rules must specify AC1_rule_variant as either 'Y_as_constant' or 'X_as_constant'")
        if self.rule_type != "AC1" and self.AC1_rule_variant is not None:
            raise ValueError("AC1_rule_variant should only be specified for C rules")
        
        flattened_nodes = self.bottom_rule.get_flattened_nodes()
        
        self.node_mappings = dict.fromkeys(flattened_nodes)
        self.node_mappings[flattened_nodes[0]] = "Y"
        self.node_mappings[flattened_nodes[1]] = "X"

        # First two nodes are mapped to "Y" and "X"
        # Rest of the nodes are numbered with A{n+1}, i.e., starting from A2
        A_index = 2
        for unique_node_index, node in enumerate(self.node_mappings):
            if unique_node_index == 0:
                self.node_mappings[node] = "Y"
            elif unique_node_index == 1:
                self.node_mappings[node] = "X"
            else:
                ### Took this part out as it is nicely handled in the next step anyways
                # if self.bottom_rule.is_cyclical and len(self.node_mappings) == unique_node_index + 1:
                #     continue
                # else:
                while f"A{A_index}" in self.node_mappings.values():
                    A_index += 1
                self.node_mappings[node] = f"A{A_index}"

        # Now, further adjusting the dictionary based on the rule type
        if self.rule_type == "C":
            # We already handled the "Y" at the beginning and end above, 
            # So, no specific handling is needed
            pass
        if self.rule_type == "AC2":
            # Replacing the first node mapping of "Y" with itself (since it should stay as a constant)
            self.node_mappings[flattened_nodes[0]] = flattened_nodes[0]
        if self.rule_type == "AC1":
            if self.bottom_rule.is_cyclical:
                if self.AC1_rule_variant == "Y_as_constant":
                    # Replacing the first mapping with itself (i.e., replacing Y wih a constant)
                    self.node_mappings[flattened_nodes[0]] = flattened_nodes[0]
                else: # else if self.AC1_rule_variant == "X_as_constant"
                    # Replacing the second mapping with itself (i.e., replacing X wih a constant)
                    self.node_mappings[flattened_nodes[1]] = flattened_nodes[1]
            else: # self.bottom_rule is not cyclical
                # Replacing the first mapping with itself (since it should stay as a constant)
                self.node_mappings[flattened_nodes[0]] = flattened_nodes[0]
                # Replacing the last mapping with itself (since it should stay as a constant)
                self.node_mappings[flattened_nodes[-1]] = flattened_nodes[-1]

        # Doing the mapping:
        self.generalized_head = Triple(
            self.node_mappings[self.bottom_rule.head.subject],
            self.bottom_rule.head.relation,
            self.node_mappings[self.bottom_rule.head.object]
        )
        self.generalized_body = [
            Triple(
                self.node_mappings[triple.subject],
                triple.relation,
                self.node_mappings[triple.object]
            )
            for triple in self.bottom_rule.body
        ]

    def calculate_confidence(self, kg: KnowledgeGraph, sample_size: int = 500, pc: float = 1.0) -> float:
        """
        Calculate approximate confidence based on sampling.
        
        :param kg: Knowledge Graph to sample from
        :param sample_size: Number of body groundings to sample
        :param pc: Pessimistic constant for confidence smoothing (Laplace-like)
        :return: Confidence score between 0 and 1
        """
        # Reset counters
        self.body_groundings_count = 0
        self.head_groundings_count = 0
        
        # Get the variables that need to be bound
        variable_bindings = self._get_variable_bindings()
        
        # Sampling body groundings
        for _ in range(sample_size):
            # Try to find a valid body grounding
            grounding = self._sample_body_grounding(kg, variable_bindings)
            if not grounding:
                continue
                
            self.body_groundings_count += 1
            
            # Check if this grounding makes the head true
            if self._check_head_grounding(kg, grounding):
                self.head_groundings_count += 1
        
        # Calculate confidence with smoothing
        if self.body_groundings_count > 0:
            self.confidence = (self.head_groundings_count + pc) / (self.body_groundings_count + pc)
        else:
            self.confidence = 0.0
            
        return self.confidence

    def _get_variable_bindings(self) -> Set[str]:
        """
        Get all variables that need to be bound when sampling.
        Excludes constants and the head variables that will be determined by the body.
        """
        variables = set()
        
        # Add body variables
        for triple in self.generalized_body:
            subj_str = str(triple.subject)
            obj_str = str(triple.object)
            
            if not subj_str.startswith('A') and not self._is_constant(subj_str):
                variables.add(subj_str)
            if not obj_str.startswith('A') and not self._is_constant(obj_str):
                variables.add(obj_str)
                
        # Add auxiliary variables (A2, A3, etc.)
        aux_vars = {str(v) for v in self.node_mappings.values() if str(v).startswith('A')}
        variables.update(aux_vars)
        
        return variables

    def _is_constant(self, value: str) -> bool:
        """Check if a value is a constant (not a variable)."""
        return not (value.startswith('A') or value in {'X', 'Y'})

    def _sample_body_grounding(self, kg: KnowledgeGraph, variables: Set[str]) -> Optional[Dict[str, Any]]:
        """
        Try to sample a valid grounding for the body.
        Returns None if no valid grounding could be found.
        """
        max_attempts = 50  # Prevent infinite loops
        for _ in range(max_attempts):
            grounding = {}
            
            # First bind any constants
            for triple in self.generalized_body:
                if self._is_constant(str(triple.subject)):
                    grounding[str(triple.subject)] = triple.subject
                if self._is_constant(str(triple.object)):
                    grounding[str(triple.object)] = triple.object
            
            # Try to bind all required variables
            success = True
            for triple in self.generalized_body:
                if not self._bind_triple_variables(kg, triple, grounding):
                    success = False
                    break
                    
            if success:
                return grounding
                
        return None

    def _bind_triple_variables(self, kg: KnowledgeGraph, triple: Triple, grounding: Dict[str, Any]) -> bool:
        """
        Try to bind variables for a single triple in the body.
        Returns False if no valid binding could be found.
        """
        subj_key = str(triple.subject)
        obj_key = str(triple.object)
        
        # Case 1: Both subject and object are already bound
        if subj_key in grounding and obj_key in grounding:
            return kg.has_fact(grounding[subj_key], triple.relation, grounding[obj_key])
        
        # Case 2: Subject is bound, object needs binding
        elif subj_key in grounding:
            possible_objects = kg.adj[triple.relation].get(grounding[subj_key], set())
            if not possible_objects:
                return False
            grounding[obj_key] = random.choice(list(possible_objects))
            return True
            
        # Case 3: Object is bound, subject needs binding
        elif obj_key in grounding:
            possible_subjects = kg.adj_inv[triple.relation].get(grounding[obj_key], set())
            if not possible_subjects:
                return False
            grounding[subj_key] = random.choice(list(possible_subjects))
            return True
            
        # Case 4: Neither is bound
        else:
            # Pick a random subject that has this relation
            if not kg.adj[triple.relation]:
                return False
            random_subject = random.choice(list(kg.adj[triple.relation].keys()))
            grounding[subj_key] = random_subject
            
            # Then pick a random object for that subject
            possible_objects = kg.adj[triple.relation][random_subject]
            if not possible_objects:
                return False
            grounding[obj_key] = random.choice(list(possible_objects))
            return True

    def _check_head_grounding(self, kg: KnowledgeGraph, grounding: Dict[str, Any]) -> bool:
        """Check if the head triple is true given a grounding."""
        head_subj_key = str(self.generalized_head.subject)
        head_obj_key = str(self.generalized_head.object)
        
        head_subj = grounding.get(head_subj_key, self.generalized_head.subject)
        head_obj = grounding.get(head_obj_key, self.generalized_head.object)
        
        return kg.has_fact(head_subj, self.generalized_head.relation, head_obj)

    def to_logical_string(self) -> str:
        # Head
        head_triple = self.bottom_rule.head
        head_str = f"{head_triple.relation}({self.node_mappings[head_triple.subject]}, {self.node_mappings[head_triple.object]})"

        # Body
        body_parts = []
        for triple in self.bottom_rule.body:
            part = f"{triple.relation}({self.node_mappings[triple.subject]}, {self.node_mappings[triple.object]})"
            body_parts.append(part)

        if body_parts:
            return f"{head_str} <- {', '.join(body_parts)}"
        else:
            return head_str

    def __str__(self) -> str:
        if self.bottom_rule is None:
            return "GeneralizedRule(None)"

        head_chain, body_chain = self.bottom_rule.get_chained()
        
        # Head
        head_mapped = (
            self.node_mappings.get(head_chain[0], head_chain[0]),
            head_chain[1],
            self.node_mappings.get(head_chain[2], head_chain[2])
        )
        
        # Body
        body_mapped = [
            (
                self.node_mappings.get(a1, a1),
                r,
                self.node_mappings.get(a2, a2)
            )
            for (a1, r, a2) in body_chain
        ]
        
        head_str = f"{head_mapped[1]}({head_mapped[0]}, {head_mapped[2]})"
        body_str = ", ".join(f"{r}({a1}, {a2})" for (a1, r, a2) in body_mapped)
        
        return f"{head_str} <- {body_str}"

def generalize_bottom_rule(bottom_rule: BottomRule) -> List[GeneralizedRule]:
    """
    From a single BottomRule instance, generate the possible rules:
       - If cyclical => [C-rule-X, C-rule-Y, AC1-rule]
       - If acyclic => [AC1-rule, AC2-rule]
    """
    if bottom_rule is None:
        print("BottomRule is None")
        return []

    if bottom_rule.is_cyclical:
        # 3 rules: C, AC1 with Y as constant, and AC1 with X as constant.
        return [
            GeneralizedRule(bottom_rule, "AC1", "Y_as_constant"),
            GeneralizedRule(bottom_rule, "AC1", "X_as_constant"),
            GeneralizedRule(bottom_rule, "C")
        ]
    else:
        # 2 rules: AC1 and AC2
        return [
            GeneralizedRule(bottom_rule=bottom_rule, rule_type="AC1"),
            GeneralizedRule(bottom_rule=bottom_rule, rule_type="AC2")
        ]
