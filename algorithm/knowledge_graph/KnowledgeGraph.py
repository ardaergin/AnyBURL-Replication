from collections import defaultdict
from .Triple import Triple

class KnowledgeGraph:
    def __init__(self, triples):
        """
        :param triples: list of Triple objects or (subject, relation, object) tuples.
        """
        self.triples = []
        self.outgoing = defaultdict(list)
        self.incoming = defaultdict(list)
        self.adj = defaultdict(lambda: defaultdict(set))
        # self.adj[relation][subject] = {object_1, object_2, ..., object_n}
        self.adj_inv = defaultdict(lambda: defaultdict(set))
        # self.adj_inv[relation][object] = {subject_1, subject_2, ..., subject_n}
        self.relations = set()
        self.entities = set()

        for triple in triples:
            if not isinstance(triple, Triple):
                triple = Triple.from_tuple(triple)
            self.triples.append(triple)
            s, r, o = triple
            self.outgoing[s].append((r, o))
            self.incoming[o].append((r, s))
            self.adj[r][s].add(o)
            self.adj_inv[r][o].add(s)
            self.relations.add(r)
            self.entities.add(s)
            self.entities.add(o)
        
        self.relations = list(self.relations)
        self.entities = list(self.entities)

    def size(self):
        """
        Return the complete list (or set) of relations (relation IDs) in the KG.
        """
        return len(self.triples)

    def get_entities(self):
        """
        Return the complete list (or set) of entity IDs in the KG.
        """
        return self.entities

    def get_relations(self):
        """
        Return the complete list (or set) of relations (relation IDs) in the KG.
        """
        return self.relations
    
    def has_fact(self, s, r, o):
        """
        Quickly check if a specific triple (s,r,o) is in the KG.
        """
        return o in self.adj[r][s]
