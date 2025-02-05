import random
from typing import Optional
from ..knowledge_graph import KnowledgeGraph, Triple
from .BottomRule import BottomRule

def pick_step_direction(direction_allowed):
    """
    Decide whether the next step is 'forward' or 'backward'.
    """
    if direction_allowed == "both":
        return 'forward' if random.random() < 0.5 else 'backward'
    elif direction_allowed == "forward-only":
        return 'forward'
    elif direction_allowed == "backward-only":
        return 'backward'
    else:
        raise ValueError(f"Unsupported direction_allowed: {direction_allowed}")


def get_possible_moves(kg: KnowledgeGraph, current_node: str, step_direction: str):
    """
    Enumerate all possible edges from 'current_node' in the chosen direction.
    
    :param kg: KnowledgeGraph instance
    :param current_node: The current entity node
    :param step_direction: 'forward' or 'backward'
    :return: List of Triples (subject, relation, object, [timestamp]).
    """
    if step_direction == 'forward':
        return [
            Triple.from_tuple((current_node, r, o))
            for (r, o) in kg.outgoing.get(current_node, [])
        ]
    else:
        return [
            Triple.from_tuple((s, r, current_node))
            for (r, s) in kg.incoming.get(current_node, [])
        ]


def filter_valid_moves(bottom_rule: BottomRule, 
                       possible_moves: list, 
                       step_direction: str, 
                       is_last_step: bool):
    """
    Filter out moves that revisit intermediate nodes unless it's the final step and cycles are allowed.

    :param bottom_rule: BottomRule instance
    :param possible_moves: List of Triples
    :param step_direction: 'forward' or 'backward'
    :param is_last_step: Boolean indicating if it's the last step
    :return: List of filtered valid moves
    """
    filtered_possible_moves = []

    for move in possible_moves:
        if step_direction == 'forward':
            next_node = move.object
            if next_node in bottom_rule.visited:
                if is_last_step:
                    if bottom_rule.start_from == 'subject' and next_node == bottom_rule.head.object:
                        filtered_possible_moves.append(move)
                    elif bottom_rule.start_from == 'object' and next_node == bottom_rule.head.subject:
                        filtered_possible_moves.append(move)
            else:
                filtered_possible_moves.append(move)
        else:
            prev_node = move.subject
            if prev_node in bottom_rule.visited:
                if is_last_step:
                    if bottom_rule.start_from == 'subject' and prev_node == bottom_rule.head.object:
                        filtered_possible_moves.append(move)
                    elif bottom_rule.start_from == 'object' and prev_node == bottom_rule.head.subject:
                        filtered_possible_moves.append(move)
            else:
                filtered_possible_moves.append(move)
    return filtered_possible_moves


def sample_bottom_rule(kg: KnowledgeGraph,
                       n: int = 2,
                       direction_allowed: str = "both",
                       temporal_window: Optional[float] = None) -> Optional[BottomRule]:
    """
    Sample a bottom-up rule (i.e., a path) of length n from the knowledge graph,
    enforcing a time ordering constraint if timestamps are present.

    :param kg: KnowledgeGraph
    :param n: total number of edges for the resulting 'bottom rule'.
              The HEAD triple counts as 1, so the BODY has (n - 1).
    :param direction_allowed: "both", "forward-only", or "backward-only"
    :param temporal_window: If not None, imposes a maximum gap between consecutive timestamps.
                            E.g. 3600 => events can't be more than an hour apart.
    :return: A BottomRule object or None if no valid path is found.
    """
    # ValueError for n
    if n < 1:
        raise ValueError("n must be >= 1")

    # -------------------
    # 1) Picking the HEAD triple
    # -------------------
    # Filter to triples with timestamps when temporal_window is set
    if temporal_window is not None:
        valid_head_triples = [t for t in kg.triples if t.timestamp is not None]
        if not valid_head_triples:
            return None
        head_triple = random.choice(valid_head_triples)
    else:
        head_triple = random.choice(kg.triples)
    
    # ---------------------------
    # 2) Deciding the 'start node'
    # ---------------------------
    initial_start_node = random.choice([head_triple.subject, head_triple.object])
    start_from = 'subject' if initial_start_node == head_triple.subject else 'object'
    current_node = initial_start_node

    # Forming the Bottom Rule
    bottom_rule = BottomRule(head_triple, start_from)

    # ---------------------------
    # 3) Expanding Bottom-rule with (length - 1) edges
    # ---------------------------

    # If length=1, no body edges needed. We're done.
    if n == 1:
        return bottom_rule
    
    # Else, loop for taking steps
    for step_id in range(n - 1): 

        # 3A) We must first decide whether we are going to go forward or backward!
        step_direction = pick_step_direction(direction_allowed)

        # 3B) Get a list of all possible moves in a (subj, rel, obj) format 
        possible_moves = get_possible_moves(kg, current_node, step_direction)
        if not possible_moves:
            return None

        # 3C) We then filter out this list to ensure straigth paths
        is_last_step = (step_id == (n - 2))
        filtered_moves = filter_valid_moves(bottom_rule, possible_moves, step_direction, is_last_step)
        if not filtered_moves:
            return None
        
        # Temporal filtering: if we have current_time in the bottom_rule,
        # we discard any moves with timestamps that go backwards or exceed the window.
        valid_moves = []
        for move in filtered_moves:
            if bottom_rule.current_time is not None and move.timestamp is not None:
                # Non-decreasing time
                if move.timestamp < bottom_rule.current_time:
                    continue
                # Checking time window
                if temporal_window is not None:
                    dt = move.timestamp - bottom_rule.current_time
                    if dt > temporal_window:
                        continue
            valid_moves.append(move)

        if not valid_moves:
            return None

        # 3D) Now, we can pick one triple from possible_moves at random, add that to our bottom-rule
        if step_direction == 'forward':
            triple = random.choice(valid_moves) # (current_node, relation, next_node)
            bottom_rule.add_triple(triple, step_direction)
            # Updating the current node: Move forward to the tail
            bottom_rule.visited.add(triple.object) # next_node
            current_node = triple.object # next_node
        else:
            triple = random.choice(valid_moves) # (prev_node, relation, current_node)
            bottom_rule.add_triple(triple, step_direction)
            # Updating the current node: Move backward to the head
            bottom_rule.visited.add(triple.subject) # prev_node
            current_node = triple.subject # prev_node

    # ---------------------------------
    # 4) Check if cyclic bottom rule, and add it as an attribute
    # ---------------------------------
    if current_node == bottom_rule.head.object or current_node == bottom_rule.head.subject:
        bottom_rule.is_cyclical = True

    return bottom_rule
