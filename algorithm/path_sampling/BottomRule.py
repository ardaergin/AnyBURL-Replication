from dataclasses import dataclass, field
from typing import List, Tuple, Set
from algorithm.knowledge_graph import KnowledgeGraph, Triple

@dataclass
class BottomRule:
    head: Triple
    start_from: str  # "subject" or "object"
    body: List[Triple] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)  # 'forward' or 'backward'
    is_cyclical: bool = False
    visited: Set[str] = field(default_factory=set)

    def __post_init__(self):
        # Initializing visited with the two nodes from the head triple.
        self.visited = {self.head.subject, self.head.object}

    def add_triple(self, triple: Triple, step: str) -> None:
        """
        Add a new edge (subj, rel, obj) to the rule's body along with its step direction.
        """
        self.body.append(triple)
        self.steps.append(step)

    def to_dict(self) -> dict:
        """
        Convert the bottom rule into a dictionary structure.
        """
        return {
            "head": tuple(self.head),
            "start_from": self.start_from,
            "body": [tuple(triple) for triple in self.body],
            "steps": self.steps,
            "is_cyclical": self.is_cyclical
        }

    def get_chained(self) -> Tuple[Tuple[str, str, str], List[Tuple[str, str, str]]]:
        """
        Returns the bottom rule in a chained (canonical) representation.
        Head:
          - If start_from == 'object', the head remains as tuple(head).
          - Otherwise, the head is flipped.
        Body:
          - For each body edge, if the step is 'forward', use tuple(edge);
            if 'backward', use tuple(edge.flipped()).
        """
        # Adjusting the Head
        head_for_chain = tuple(self.head) if self.start_from == 'object' else tuple(self.head.flipped())

        # Adjusting the Body
        body_for_chain = []
        for step, edge in zip(self.steps, self.body):
            if step == 'forward':
                body_for_chain.append(tuple(edge))
            else:
                body_for_chain.append(tuple(edge.flipped()))

        return head_for_chain, body_for_chain

    def get_flattened_nodes(self):
        """
        Returns a flattened list of nodes in the BottomRule, 
        preserving order from the chained representation.
        Duplicates are allowed.
        """
        flattened_nodes: List[str] = []
        chained_head, chained_body = self.get_chained()

        # Head
        flattened_nodes.extend([chained_head[0], chained_head[2]])

        # Body
        for a1, _, a2 in chained_body:
            flattened_nodes.extend([a1, a2])

        return flattened_nodes

    def __str__(self) -> str:
        head_chain, body_chain = self.get_chained()

        # Constructing the head string:
        (head_Y, head_R, head_X) = head_chain
        head_str = f"{head_R}({head_Y}, {head_X})"

        # Constructing the body string:
        body_str_parts = [f"{r}({a1}, {a2})" for (a1, r, a2) in body_chain]
        body_str = ", ".join(body_str_parts)

        return f"{head_str} <- {body_str}"

    def __repr__(self) -> str:
        return (f"BottomRule(head={self.head!r}, start_from={self.start_from!r}, "
                f"body={[triple for triple in self.body]!r}, steps={self.steps!r}, "
                f"is_cyclical={self.is_cyclical!r})")
