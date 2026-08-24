"""Evidence aggregation.

Turns retrieved records + an independent coverage report into a structured evidence
bundle, and synthesises **gap evidence**: an unobserved interval is itself a first-class
piece of evidence, not the silent absence of one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.coverage.ledger import CoverageLedger
from app.evidence.classifier import classify, in_window, region_relevant
from app.models.schemas import (
    CoverageReport,
    Evidence,
    EvidenceState,
    Modality,
    QueryPlan,
    RetrievedDoc,
    SourceRecord,
    TimeRange,
)
from app.retrieval.corpus import Corpus


def base_state(evidence: Evidence) -> EvidenceState:
    """The epistemic state before CONTRADICTION was layered on top of it."""
    raw = evidence.attributes.get("base_state")
    if raw:
        try:
            return EvidenceState(raw)
        except ValueError:
            pass
    return evidence.state


def matches_entities(evidence: Evidence, entities: list[str]) -> bool:
    if not entities:
        return True
    haystack = " ".join(
        [evidence.claim.lower()]
        + [e.lower() for e in evidence.entities]
        + [
            str(evidence.attributes.get(k, "")).lower()
            for k in ("track_id", "vessel_name", "mmsi")
        ]
    )
    return any(e.lower() in haystack for e in entities)


@dataclass
class EvidenceBundle:
    evidence: list[Evidence] = field(default_factory=list)
    coverage: CoverageReport | None = None
    #: When the operator names specific modalities or entities, a claim about *those*
    #: may only be supported by evidence of that kind. Retrieval stays soft (spec 8);
    #: the claim does not. Answering "yes, AIS contacts" on the strength of a radar track
    #: would be a different kind of fabrication.
    scope_modalities: list[Modality] = field(default_factory=list)
    scope_entities: list[str] = field(default_factory=list)

    def in_scope(self, evidence: Evidence) -> bool:
        if self.scope_modalities and evidence.source not in self.scope_modalities:
            return False
        return matches_entities(evidence, self.scope_entities)

    # -------------------------------------------------------------------- selectors ----
    def by_state(self, *states: EvidenceState) -> list[Evidence]:
        return [e for e in self.evidence if e.state in states]

    @property
    def contradictory(self) -> list[Evidence]:
        return [e for e in self.evidence if e.state is EvidenceState.CONTRADICTION]

    @property
    def operational(self) -> list[Evidence]:
        """Evidence admissible for an operational claim: in-window and in-region."""
        return [
            e
            for e in self.evidence
            if e.attributes.get("in_window")
            and e.attributes.get("region_relevant")
            and e.state not in (EvidenceState.STALE,)
        ]

    @property
    def presence(self) -> list[Evidence]:
        return [
            e
            for e in self.operational
            if base_state(e) is EvidenceState.PRESENCE
            and e.source not in (Modality.STANDING_ORDER, Modality.TERRAIN)
            and e.attributes.get("detection", False)
            and self.in_scope(e)
        ]

    @property
    def absence(self) -> list[Evidence]:
        return [e for e in self.operational if base_state(e) is EvidenceState.OBSERVED_ABSENCE]

    @property
    def partial(self) -> list[Evidence]:
        return [e for e in self.operational if base_state(e) is EvidenceState.PARTIAL_COVERAGE]

    @property
    def unobserved(self) -> list[Evidence]:
        # Region relevance is enforced here too: a negative report from a *different*
        # region says nothing about the queried one, however similar its wording.
        items = [
            e
            for e in self.evidence
            if base_state(e) is EvidenceState.UNOBSERVED
            and e.attributes.get("region_relevant", True)
        ]
        # Ledger-derived gaps first: for an UNKNOWN answer the gap is the finding, and it
        # is what should be cited before any rejected negative report.
        return sorted(items, key=lambda e: 0 if e.attributes.get("kind") else 1)

    @property
    def stale(self) -> list[Evidence]:
        return [e for e in self.evidence if e.state is EvidenceState.STALE]

    @property
    def low_confidence(self) -> list[Evidence]:
        return [e for e in self.evidence if e.state is EvidenceState.LOW_CONFIDENCE]

    @property
    def context(self) -> list[Evidence]:
        return [
            e for e in self.evidence if e.source in (Modality.STANDING_ORDER, Modality.TERRAIN)
        ]

    def get(self, evidence_id: str) -> Evidence | None:
        for e in self.evidence:
            if e.evidence_id == evidence_id:
                return e
        return None

    def ids(self) -> set[str]:
        return {e.evidence_id for e in self.evidence}


def _gap_evidence(report: CoverageReport) -> list[Evidence]:
    """Materialise coverage gaps as UNOBSERVED evidence records."""
    out: list[Evidence] = []
    for i, (start, end) in enumerate(report.missing_intervals, start=1):
        blind = ", ".join(report.blind_subregions) or report.region
        out.append(
            Evidence(
                evidence_id=f"GAP-{i:02d}",
                source_id="coverage_ledger",
                source=Modality.TERRAIN,
                sensor="coverage_ledger",
                claim=(
                    f"No sensing coverage of {blind} between "
                    f"{start.strftime('%H:%M')} and {end.strftime('%H:%M')}Z. "
                    "Nothing can be asserted about this interval."
                ),
                state=EvidenceState.UNOBSERVED,
                region=blind,
                time_range=TimeRange(start=start, end=end),
                reliability=1.0,
                recency=1.0,
                attributes={
                    "base_state": EvidenceState.UNOBSERVED.value,
                    "in_window": True,
                    "region_relevant": True,
                    "kind": "coverage_gap",
                    "ledger_entries": report.ledger_entries[:8],
                },
                notes=["derived from the coverage ledger, not from retrieval"],
            )
        )
    for j, modality in enumerate(report.missing_modalities, start=len(out) + 1):
        out.append(
            Evidence(
                evidence_id=f"GAP-{j:02d}",
                source_id="coverage_ledger",
                source=modality,
                sensor="coverage_ledger",
                claim=(
                    f"No {modality.value.replace('_', ' ')} coverage of {report.region} "
                    f"during {report.time_range.label()}Z."
                ),
                state=EvidenceState.UNOBSERVED,
                region=report.region,
                time_range=report.time_range,
                reliability=1.0,
                recency=1.0,
                attributes={
                    "base_state": EvidenceState.UNOBSERVED.value,
                    "in_window": True,
                    "region_relevant": True,
                    "kind": "missing_modality",
                },
                notes=["derived from the coverage ledger, not from retrieval"],
            )
        )
    return out


def aggregate(
    plan: QueryPlan,
    docs: list[RetrievedDoc],
    corpus: Corpus,
    ledger: CoverageLedger,
    coverage: CoverageReport,
) -> EvidenceBundle:
    bundle = EvidenceBundle(
        coverage=coverage,
        scope_modalities=list(plan.preferred_modalities) if plan.modalities_explicit else [],
        scope_entities=list(plan.entities),
    )
    for doc in docs:
        record: SourceRecord | None = corpus.get(doc.record_id)
        if record is None:
            continue
        bundle.evidence.append(
            classify(
                record=record,
                ledger=ledger,
                window=plan.time_range,
                query_region=plan.region,
                retrieval_score=doc.score,
            )
        )
    bundle.evidence.extend(_gap_evidence(coverage))
    return bundle
