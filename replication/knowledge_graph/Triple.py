from dataclasses import dataclass, field

@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    object: str
    reversed: bool = field(default=False, compare=False)

    def flipped(self) -> "Triple":
        """
        Return a new Triple with subject and object swapped.
        The 'reversed' flag is toggled.
        """
        return Triple(subject=self.object, relation=self.relation, object=self.subject, reversed=not self.reversed)

    @classmethod
    def from_tuple(cls, triple_tuple, reversed=False):
        return cls(triple_tuple[0], triple_tuple[1], triple_tuple[2], reversed=reversed)

    def __iter__(self):
        """
        So, tuple(my_triple) would directly give (my_triple.subject, my_triple.relation, my_triple.object)
        """
        yield self.subject
        yield self.relation
        yield self.object

    def __str__(self):
        """
        String representation for printing, including whether the triple is reversed.
        """
        base = f"({self.subject}, {self.relation}, {self.object})"
        return f"{base} [reversed]" if self.reversed else base

    def __repr__(self):
        return (
        f"Triple(subject={self.subject!r}, "
        f"relation={self.relation!r}, "
        f"object={self.object!r}, "
        f"reversed={self.reversed!r})"
        )
