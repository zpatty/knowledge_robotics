"""Reference classification for B2 — is a cited work inside robotics or outside?

`B2` (docs/02 section 6) is the fraction of a technique's cited support lying
outside the robotics literature, plus the age distribution of those citations. It
is the only epistemic-base indicator computable from metadata alone, so it carries
H5 (docs/04 section 4).

**DOI structure is the primary signal, venue text the fallback.** Measured on the
corpus: 69% of Crossref reference entries carry a DOI but only 29% carry a venue
string, so classifying on venue text alone reaches about a tenth of the reference
list. IEEE DOIs additionally encode the venue in their suffix — `10.1109/ICRA.2015`
and `10.1109/ICRA40945.2020` both reduce to `ICRA` — and IEEE is 56% of all
DOI-bearing references here.

All mappings live in `registry/reference_classes.json`: hand-written, checkable by
eye, editable without touching code. Nothing is inferred.

**Unknown is a reported category, not a default.** Treating unmatched references as
external would inflate reference reach exactly where metadata is thinnest, which is
the early corpus, and would manufacture a trend. `reach` is therefore computed over
*classified* references only and always travels with `classified_share`, so a reader
can see how much of the list the number rests on.

Reference *age* needs no classification and is the more robust half of B2: the gap
between citing and cited year. docs/01 section 1.4 argues a wide base cites old and
external, a narrow one recent and internal.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Mapping

from .config import ROOT

DEFAULT_CLASSES = ROOT / "registry" / "reference_classes.json"

_WS = re.compile(r"\s+")
_STEM = re.compile(r"^([A-Za-z]+|\d+)")
_YEAR = re.compile(r"(1[89]\d{2}|20\d{2})")


class ReferenceClassifier:
    """Classify a reference as robotics / external / unknown, with its field."""

    def __init__(self, path: Path | None = None):
        spec = json.loads((path or DEFAULT_CLASSES).read_text())
        # Flatten to lookup tables; the JSON is grouped for human editing, and
        # the inversion happens once here rather than per reference.
        self.stem_class: dict[str, str] = {}
        for field, stems in spec["ieee_stems"].items():
            for stem in stems:
                self.stem_class[stem.upper()] = field
        self.registrant_class: dict[str, str] = {}
        for field, prefixes in spec["registrant_prefixes"].items():
            for prefix in prefixes:
                self.registrant_class[prefix] = field
        self.doi_prefix_class: list[tuple[str, str]] = [
            (prefix.lower(), field)
            for field, prefixes in spec["doi_prefixes"].items()
            for prefix in prefixes
        ]
        self.venue_markers: list[tuple[str, str]] = [
            (marker.lower(), field)
            for field, markers in spec["venue_markers"].items()
            for marker in markers
        ]

    @staticmethod
    def ieee_stem(doi: str) -> str | None:
        """`10.1109/ICRA40945.2020.9196733` -> `ICRA`. None if not an IEEE DOI.

        The trailing digits are a per-conference-instance id that changed format
        around 2019; stripping them merges `ICRA`, `ICRA40945` and `ICRA48506`
        into one venue, which is what they are.
        """
        if not doi.lower().startswith("10.1109/"):
            return None
        suffix = doi.split("/", 1)[1]
        match = _STEM.match(suffix)
        return match.group(1).upper() if match else None

    def classify(self, reference: Mapping) -> tuple[str, str | None]:
        """Return (class, field): class is 'robotics' | 'external' | 'unknown'."""
        doi = (reference.get("doi") or "").lower()
        if doi:
            stem = self.ieee_stem(doi)
            if stem and stem in self.stem_class:
                return self._as_class(self.stem_class[stem])
            for prefix, field in self.doi_prefix_class:
                if doi.startswith(prefix):
                    return self._as_class(field)
            registrant = doi.split("/", 1)[0]
            if registrant in self.registrant_class:
                return self._as_class(self.registrant_class[registrant])

        venue = _WS.sub(" ", (reference.get("venue") or "").lower())
        if venue:
            # Robotics first: a robotics venue whose name contains a general word
            # like "science" must not be read as external.
            for marker, field in self.venue_markers:
                if field == "robotics" and marker in venue:
                    return "robotics", None
            for marker, field in self.venue_markers:
                if field != "robotics" and marker in venue:
                    return "external", field
        return "unknown", None

    @staticmethod
    def _as_class(field: str) -> tuple[str, str | None]:
        return ("robotics", None) if field == "robotics" else ("external", field)


def reference_year(reference: Mapping) -> int | None:
    raw = reference.get("year")
    if raw is None:
        return None
    match = _YEAR.search(str(raw))
    return int(match.group(1)) if match else None


def summarise(
    references: Iterable[Mapping],
    citing_year: int | None,
    classifier: ReferenceClassifier,
) -> dict:
    """B2 components for one paper. All ratios, no raw counts as outputs (docs/09)."""
    counts = {"robotics": 0, "external": 0, "unknown": 0}
    fields: dict[str, int] = {}
    ages: list[int] = []
    total = 0

    for ref in references:
        total += 1
        cls, field = classifier.classify(ref)
        counts[cls] += 1
        if field:
            fields[field] = fields.get(field, 0) + 1
        year = reference_year(ref)
        if year and citing_year and 0 <= citing_year - year <= 100:
            ages.append(citing_year - year)

    classified = counts["robotics"] + counts["external"]
    ages.sort()
    return {
        "n_references": total,
        "n_robotics": counts["robotics"],
        "n_external": counts["external"],
        "n_unknown": counts["unknown"],
        "classified_share": classified / total if total else 0.0,
        "reach": counts["external"] / classified if classified else None,
        "external_fields": fields,
        "n_aged": len(ages),
        "age_mean": sum(ages) / len(ages) if ages else None,
        "age_median": ages[len(ages) // 2] if ages else None,
    }
