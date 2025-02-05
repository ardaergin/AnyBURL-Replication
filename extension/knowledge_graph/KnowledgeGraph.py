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
        self.time_index = defaultdict(list) # <--------
        self.adj_by_time = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        self.adj_inv_by_time = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        self.relations = set()
        self.entities = set()

        for triple in triples:
            if not isinstance(triple, Triple):
                triple = Triple.from_tuple(triple)
            self.triples.append(triple)

            s, r, o = triple.subject, triple.relation, triple.object
            self.outgoing[s].append((r, o))
            self.incoming[o].append((r, s))
            self.adj[r][s].add(o)
            self.adj_inv[r][o].add(s)
            
            if triple.timestamp is not None:
                t = triple.timestamp
                self.time_index[t].append(triple)
                self.adj_by_time[t][r][s].add(o)
                self.adj_inv_by_time[t][r][o].add(s)

            self.relations.add(r)
            self.entities.add(s)
            self.entities.add(o)
        
        self.relations = list(self.relations)
        self.entities = list(self.entities)

    def size(self):
        """
        Return the number of triples in the KG.
        """
        return len(self.triples)

    def get_entities(self):
        """
        Return the complete list of entity IDs in the KG.
        """
        return self.entities

    def get_relations(self):
        """
        Return the complete list of relations in the KG.
        """
        return self.relations

    def has_fact(self, s, r, o):
        """
        Quickly check if a specific triple (s, r, o) is in the KG (ignoring time).
        """
        return o in self.adj[r][s]

    # ---------------------
    # Time-indexed queries:
    # ---------------------
    def get_triples_at_time(self, timestamp):
        """
        Return a list of all triples that have exactly this timestamp.
        """
        return self.time_index.get(timestamp, [])

    def get_triples_in_interval(self, start_time, end_time):
        """
        Return a list of all triples whose timestamps fall in [start_time, end_time].
        Note: This naive approach loops over ALL timestamps. If you have a very
        large number of distinct timestamps, consider storing them in a sorted structure
        and performing a binary search-based range query instead.
        """
        result = []
        for t in self.time_index:
            if start_time <= t <= end_time:
                result.extend(self.time_index[t])
        return result

    def has_fact_temporal(self, s, r, o, timestamp=None, tolerance=0):
        """
        Check if (s, r, o) is valid at the given timestamp (with an optional +/- tolerance).
        
        If timestamp is None, we fall back to a plain 'has_fact' check.
        Otherwise, we do a range search over [timestamp - tolerance, timestamp + tolerance].
        
        For integer timestamps, you can iterate from int(timestamp - tol) to int(timestamp + tol).
        For float timestamps, you might do a naive loop over all distinct times. 
        """
        if timestamp is None:
            return self.has_fact(s, r, o)

        # If tolerance=0, we only check the exact timestamp adjacency
        # If tolerance>0, we do a naive range check over all times in time_index
        # Note that this can be expensive if we have many distinct timestamps.
        t_min = timestamp - tolerance
        t_max = timestamp + tolerance
        
        # Storing sorted timestamps in a list for efficient range queries:
        for t in self.time_index:
            if t_min <= t <= t_max:
                if o in self.adj_by_time[t][r][s]:
                    return True
        return False
