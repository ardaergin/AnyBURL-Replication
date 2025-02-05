from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
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
    
    # (Approximate) Confidence score
    confidence: Optional[float] = None

    def __post_init__(self):
        if self.bottom_rule is None:
            self.node_mappings = {}
            self.generalized_head = None
            self.generalized_body = []
            self.confidence = None
            return

        # Validate rule type and C_rule_with combination
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
