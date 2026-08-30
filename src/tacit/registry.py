"""Technique registry — match papers to techniques by title substring.

Deliberately the simplest thing that works, per docs/11. A technique is a name
and a list of aliases in `data/techniques.json`; a paper mentions the technique
if any alias appears in its lowercased title. No clustering, no classifier, no
embeddings. Every assignment is reproducible by eye and by grep.

What this buys: the registry is auditable, editable by a domain expert without
touching code, and stable across reruns — which the four-decade instrument
constancy argument (docs/02 section 0) demands.

What it costs, stated plainly: **recall is bounded by titling convention.** A
paper that uses ICP without saying so in its title is invisible here. The corpus
has no abstracts (docs/08 F5), so there is no cheap way to do better right now,
and no way to measure the miss rate either. Precision is checkable by hand and
should be; recall is not. Treat technique paper counts as a lower bound and never
as a census.

Matching is on a space-padded title so that short aliases (" mpc ", " uav ")
match words rather than substrings of other words.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Mapping

from .config import ROOT

DEFAULT_REGISTRY = ROOT / "registry" / "techniques.json"


class Technique:
    __slots__ = ("id", "name", "axis", "aliases", "exclude")

    def __init__(self, spec: Mapping):
        self.id: str = spec["id"]
        self.name: str = spec["name"]
        self.axis: str = spec.get("axis", "unspecified")
        self.aliases: list[str] = [a.lower() for a in spec["aliases"]]
        self.exclude: list[str] = [e.lower() for e in spec.get("exclude", [])]

    def matches(self, padded_title: str) -> bool:
        if any(bad in padded_title for bad in self.exclude):
            return False
        return any(alias in padded_title for alias in self.aliases)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Technique {self.id}>"


def load_registry(path: Path | None = None) -> list[Technique]:
    data = json.loads((path or DEFAULT_REGISTRY).read_text())
    return [Technique(t) for t in data["techniques"]]


_WS = re.compile(r"\s+")


def normalise_title(title: str | None) -> str:
    """Lowercase, collapse whitespace, and pad with spaces.

    The padding lets a short alias be written as " mpc " and match the word at
    either end of the title, without a word-boundary regex per alias.
    """
    if not title:
        return " "
    return " " + _WS.sub(" ", title.lower().strip()) + " "


def match_paper(paper: Mapping, registry: Iterable[Technique]) -> list[str]:
    """Technique ids mentioned in this paper's title. May be empty or several.

    A paper matching several techniques is kept in all of them: techniques
    genuinely co-occur, and dropping or arbitrarily assigning such papers would
    be exactly the kind of cut point docs/10 rules out.
    """
    padded = normalise_title(paper.get("title"))
    return [t.id for t in registry if t.matches(padded)]
