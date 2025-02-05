from dataclasses import dataclass, field
from typing import Optional

@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    object: str
    # any float representation or UNIX time:
    timestamp: Optional[float] = None   # <---- 
    reversed: bool = field(default=False, compare=False)

    def flipped(self) -> "Triple":
        return Triple(
            subject=self.object,
            relation=self.relation,
            object=self.subject,
            timestamp=self.timestamp,  # <----
            reversed=not self.reversed
        )

    @classmethod
    def from_tuple(cls, triple_tuple, timestamp: Optional[float] = None, reversed=False):
        if len(triple_tuple) == 4:
            return cls(triple_tuple[0], triple_tuple[1], triple_tuple[2],
                       timestamp=triple_tuple[3], reversed=reversed)
        return cls(triple_tuple[0], triple_tuple[1], triple_tuple[2], timestamp=timestamp, reversed=reversed)

    def __iter__(self):
        yield self.subject
        yield self.relation
        yield self.object

    def __str__(self):
        base = f"({self.subject}, {self.relation}, {self.object}"
        if self.timestamp is not None:
            base += f", {self.timestamp})"
        else:
            base += ")"
        return f"{base} [reversed]" if self.reversed else base

    def __repr__(self):
        return (f"Triple(subject={self.subject!r}, relation={self.relation!r}, "
                f"object={self.object!r}, timestamp={self.timestamp!r}, reversed={self.reversed!r})")
