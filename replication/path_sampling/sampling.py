import random
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


def get_possible_moves(kg, current_node, step_direction):
    """
    Enumerate all possible edges from 'current_node' in the chosen direction.
    
    :param kg: KnowledgeGraph instance
    :param current_node: The current entity node
    :param step_direction: 'forward' or 'backward'
    :return: List of possible moves as tuples (subj, rel, obj).
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
    
    ### OLD CODE: ###
    # We first want to create a list of possible moves that we can take from our current_node.
    # For this purpose, we can first see all the relations a node has,
    # then, for each relation, append all the neighbors with that relation.
    # possible_moves = []
    # # Forward edges: subject -> object
    # if step_direction == 'forward':
    #     for relation in kg.adj:
    #         if current_node in kg.adj[relation]:
    #             for next_node in kg.adj[relation][current_node]:
    #                 possible_moves.append((current_node, relation, next_node))
    # # Backward edges: object -> subject 
    # else: # if step_direction == 'backward'
    #     for relation in kg.adj_inv:
    #         if current_node in kg.adj_inv[relation]:
    #             for prev_node in kg.adj_inv[relation][current_node]:
    #                 possible_moves.append((prev_node, relation, current_node))
    # return possible_moves


def filter_valid_moves(bottom_rule, possible_moves, step_direction, is_last_step):
    """
    Filter out moves that revisit intermediate nodes unless it's the final step and cycles are allowed.

    :param bottom_rule: BottomRule instance
    :param possible_moves: List of possible moves as tuples (subj, rel, obj)
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


def sample_bottom_rule(kg: KnowledgeGraph, n: int = 2, direction_allowed: str = "both"):
    """
    :param kg: KnowledgeGraph
    :param n: total number of edges for the resulting 'bottom rule'.
                The HEAD triple counts as 1, so the BODY has (n-1).
    :param direction_allowed: options are "both", "forward-only", "backward-only"
    :return: BottomRule object or None
    """
    # ValueError for n
    if n < 1:
        raise ValueError("n must be >= greater than or equal to 1")

    # -------------------
    # 1) Picking the HEAD triple
    # -------------------
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
        
        # 3D) Now, we can pick one triple from possible_moves at random, add that to our bottom-rule
        if step_direction == 'forward':
            triple = random.choice(filtered_moves) # (current_node, relation, next_node)
            bottom_rule.add_triple(triple, step_direction)
            # Updating the current node: Move forward to the tail
            bottom_rule.visited.add(triple.object) # next_node
            current_node = triple.object # next_node
        else:
            triple = random.choice(filtered_moves) # (prev_node, relation, current_node)
            bottom_rule.add_triple(triple, step_direction)
            # Updating the current node: Move backward to the head
            bottom_rule.visited.add(triple.subject) # prev_node
            current_node = triple.subject # prev_node

    # ---------------------------------
    # 4) Check if cyclic bottom rule, and add it as an attribute
    # ---------------------------------
    # if bottom_rule.start_from == 'subject':
    #     if current_node == bottom_rule.head.object:
    #         bottom_rule.is_cyclical = True
    # else:
    #     if current_node == bottom_rule.head.subject:
    #         bottom_rule.is_cyclical = True
    if current_node == bottom_rule.head.object or current_node == bottom_rule.head.subject:
        bottom_rule.is_cyclical = True

    # Return the bottom rule of length = (1) head + (length-1) body
    return bottom_rule
