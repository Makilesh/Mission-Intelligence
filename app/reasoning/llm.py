"""LLM reasoning layer.

The LLM runs **after** every deterministic decision has already been made: the answer
state, the coverage numbers, the contradictions and the confidence are fixed before the
model is called. Its job is semantic phrasing and evidence synthesis, nothing else.

Three providers:

* ``deterministic`` (default) - a template synthesiser. No network, no variance, and
  structurally incapable of inventing evidence. This is what the evaluation harness runs
  against so that retrieval/coverage metrics are not contaminated by model drift.
* ``anthropic`` / ``openai`` - real models, used with a strict system prompt.

Whatever the provider, the output passes through :func:`validate_grounding`, which rejects
any answer that cites an evidence ID that was not supplied.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from app.config import SETTINGS
from app.models.schemas import (
    AnswerState,
    Contradiction,
    CoverageReport,
    Evidence,
    QueryIntent,
    QueryPlan,
)

SYSTEM_PROMPT = """You are an evidence-grounded mission intelligence assistant.

You may only make claims supported by the supplied evidence.

An empty retrieval result does not mean absence.

If the relevant region or time window was not sufficiently observed, state that the answer
is unknown.

If sources contradict one another, explicitly report the contradiction. Do not resolve a
contradiction by preferring the more numerous or higher-scoring source.

Never manufacture sensor observations.

Every operational claim must reference one or more evidence IDs in square brackets, e.g.
[RADAR-104]. Use only the evidence IDs provided.

The ANSWER STATE, COVERAGE figures and CONFIDENCE have already been computed
deterministically. Do not contradict them, restate them as your own judgement, or offer a
different confidence value. Write 3-6 sentences of operator-facing prose."""

_ID_PATTERN = re.compile(r"\[([A-Z][A-Z0-9\-]{2,})\]")


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    grounded: bool
    violations: list[str]
    fallback_used: bool = False


# --------------------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------------------
def build_context(
    plan: QueryPlan,
    state: AnswerState,
    coverage: CoverageReport | None,
    evidence: list[Evidence],
    contradictions: list[Contradiction],
    confidence: float,
    gaps: list[str],
) -> str:
    lines: list[str] = [f"OPERATOR QUESTION: {plan.raw_question}", ""]
    lines.append(f"DETERMINED ANSWER STATE: {state.value}")
    lines.append(f"DETERMINED CONFIDENCE: {confidence:.2f}")
    lines.append("")
    if coverage:
        lines.append("OBSERVATION COVERAGE (computed independently of retrieval):")
        lines.append(f"  region: {coverage.region}")
        lines.append(f"  time range: {coverage.time_range.label()}Z")
        lines.append(f"  status: {coverage.status.value}")
        lines.append(f"  effective covered fraction: {coverage.covered_fraction:.2f}")
        lines.append(
            f"  modalities contributing: "
            f"{', '.join(m.value for m in coverage.requested_modalities) or 'none'}"
        )
        if coverage.missing_modalities:
            lines.append(
                f"  modalities with NO coverage: "
                f"{', '.join(m.value for m in coverage.missing_modalities)}"
            )
        if coverage.missing_intervals:
            intervals = ", ".join(
                f"{s.strftime('%H:%M')}-{e.strftime('%H:%M')}Z"
                for s, e in coverage.missing_intervals
            )
            lines.append(f"  unobserved intervals: {intervals}")
        if not coverage.absence_claim_supported:
            lines.append(f"  ABSENCE CLAIM BLOCKED: {coverage.absence_block_reason}")
        lines.append("")

    lines.append("EVIDENCE:")
    for e in evidence:
        lines.append(
            f"  [{e.evidence_id}] ({e.source.value}/{e.sensor}, {e.region}, "
            f"{e.time_range.start.strftime('%H:%M')}Z, state={e.state.value}, "
            f"reliability={e.reliability:.2f}) {e.claim}"
        )
    if not evidence:
        lines.append("  (none)")
    lines.append("")

    if contradictions:
        lines.append("CONTRADICTIONS (do not resolve these):")
        for c in contradictions:
            claim_text = "; ".join(
                f"[{cl.evidence_id}] {cl.source.value} says {cl.value}" for cl in c.claims
            )
            lines.append(
                f"  {c.contradiction_id} ({c.dimension.value}, severity {c.severity:.2f} "
                f"= {c.severity_label}): {c.reason}. Claims: {claim_text}"
            )
        lines.append("")

    if gaps:
        lines.append("GAPS:")
        for g in gaps:
            lines.append(f"  - {g}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Grounding validation
# --------------------------------------------------------------------------------------
def validate_grounding(text: str, allowed_ids: set[str]) -> tuple[bool, list[str]]:
    cited = {m.group(1) for m in _ID_PATTERN.finditer(text)}
    unknown = sorted(cited - allowed_ids)
    return (not unknown), [f"cited unknown evidence id {u}" for u in unknown]


# --------------------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------------------
def _cite(items: list[Evidence], limit: int = 4) -> str:
    return " ".join(f"[{e.evidence_id}]" for e in items[:limit])


def deterministic_answer(
    plan: QueryPlan,
    state: AnswerState,
    coverage: CoverageReport | None,
    evidence: list[Evidence],
    contradictions: list[Contradiction],
    confidence: float,
    presence: list[Evidence],
    absence: list[Evidence],
    unobserved: list[Evidence],
    association: str | None = None,
) -> str:
    """Template synthesis. Every sentence is derived from a structured fact."""
    region_label = (coverage.region if coverage else plan.region) or "the requested area"
    window = coverage.time_range.label() if coverage else (
        plan.time_range.label() if plan.time_range else "the mission window"
    )
    pct = f"{coverage.covered_fraction:.0%}" if coverage else "unknown"
    parts: list[str] = []

    if state is AnswerState.CONTRADICTION:
        top = contradictions[0]
        claim_text = "; ".join(
            f"{cl.source.value.replace('_', ' ')} reports {cl.value} [{cl.evidence_id}]"
            for cl in top.claims
        )
        parts.append(
            f"The sources disagree and the system cannot confidently reconcile them: {claim_text}."
        )
        parts.append(f"Disagreement dimension: {top.dimension.value} ({top.reason}).")
        if len(contradictions) > 1:
            others = ", ".join(
                f"{c.dimension.value} ({c.severity_label})" for c in contradictions[1:]
            )
            parts.append(f"Further disagreements were detected on: {others}.")
        parts.append(
            f"Observation coverage of {region_label} for {window}Z was {pct}; no source is "
            "treated as authoritative on the strength of retrieval score or source count."
        )
    elif state is AnswerState.PRESENCE:
        modalities = sorted({e.source.value.replace('_', ' ') for e in presence})
        if plan.intent is QueryIntent.ASSOCIATION:
            parts.append(
                "Association assessment follows; it is based on kinematic comparison and "
                "observation custody, not on retrieval score."
            )
        elif plan.intent is QueryIntent.ABSENCE_CHECK:
            parts.append(
                f"No - that conclusion is not supported. {len(presence)} observation(s) in "
                f"{region_label} during {window}Z report contacts "
                f"({', '.join(modalities)}). {_cite(presence)}"
            )
            if coverage and not coverage.absence_claim_supported:
                parts.append(
                    f"Independently of that, coverage was only {pct} "
                    f"({coverage.absence_block_reason}), so an absence claim could not have "
                    "been supported even if nothing had been detected."
                )
        else:
            parts.append(
                f"Yes - {len(presence)} observation(s) in {region_label} during {window}Z report "
                f"contacts, corroborated across {', '.join(modalities)}. {_cite(presence)}"
            )
        if plan.intent is not QueryIntent.ASSOCIATION:
            detail = presence[0]
            parts.append(f"Most relevant: {detail.claim} [{detail.evidence_id}]")
        parts.append(f"Observation coverage for the queried volume was {pct}.")
    elif state is AnswerState.OBSERVED_ABSENCE:
        parts.append(
            f"No contacts were observed in {region_label} during {window}Z. The coverage "
            f"ledger independently confirms {pct} effective coverage of that volume, so this "
            "is an observed absence rather than a lack of data."
        )
        if absence:
            parts.append(f"Negative reports: {_cite(absence)}")
        if coverage and coverage.missing_intervals:
            gap_text = ", ".join(
                f"{s.strftime('%H:%M')}-{e.strftime('%H:%M')}Z"
                for s, e in coverage.missing_intervals
            )
            parts.append(
                f"Caveat: {gap_text} was not observed in "
                f"{', '.join(coverage.blind_subregions) or 'part of the area'}; no claim is "
                "made about that interval."
            )
    else:  # UNKNOWN
        parts.append(
            f"Cannot determine. The sensor suite did not sufficiently observe {region_label} "
            f"during {window}Z: effective coverage was {pct}."
        )
        if coverage and coverage.absence_block_reason:
            parts.append(f"Reason: {coverage.absence_block_reason}.")
        if unobserved:
            parts.append(f"Coverage evidence: {_cite(unobserved)}")
        parts.append(
            "This is explicitly NOT a statement that nothing was present - it is a statement "
            "that the system did not look, or did not look well enough, to know."
        )
        if coverage and coverage.missing_modalities:
            parts.append(
                "Modalities with no coverage over this window: "
                + ", ".join(m.value for m in coverage.missing_modalities)
                + "."
            )

    if association:
        parts.append(association)

    if contradictions and state is not AnswerState.CONTRADICTION:
        top = contradictions[0]
        parts.append(
            f"Note: a {top.severity_label}-severity {top.dimension.value} disagreement was "
            f"also detected ({top.contradiction_id})."
        )
    return " ".join(parts)


def _call_anthropic(system: str, user: str) -> str:  # pragma: no cover - network
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model=SETTINGS.llm.model,
        max_tokens=SETTINGS.llm.max_tokens,
        temperature=SETTINGS.llm.temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in message.content if block.type == "text")


def _call_openai(system: str, user: str) -> str:  # pragma: no cover - network
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=SETTINGS.llm.model,
        temperature=SETTINGS.llm.temperature,
        max_tokens=SETTINGS.llm.max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content or ""


def synthesise(
    plan: QueryPlan,
    state: AnswerState,
    coverage: CoverageReport | None,
    evidence: list[Evidence],
    contradictions: list[Contradiction],
    confidence: float,
    gaps: list[str],
    presence: list[Evidence],
    absence: list[Evidence],
    unobserved: list[Evidence],
    association: str | None = None,
) -> LLMResult:
    provider = SETTINGS.llm.provider
    allowed = {e.evidence_id for e in evidence}
    baseline = deterministic_answer(
        plan,
        state,
        coverage,
        evidence,
        contradictions,
        confidence,
        presence,
        absence,
        unobserved,
        association,
    )

    if provider == "deterministic":
        grounded, violations = validate_grounding(baseline, allowed)
        return LLMResult(
            text=baseline,
            provider="deterministic",
            model="template-synthesis",
            grounded=grounded,
            violations=violations,
        )

    context = build_context(plan, state, coverage, evidence, contradictions, confidence, gaps)
    if association:
        context += (
            "\nDETERMINISTIC ASSOCIATION ANALYSIS (do not contradict):\n  " + association + "\n"
        )
    try:
        text = (
            _call_anthropic(SYSTEM_PROMPT, context)
            if provider == "anthropic"
            else _call_openai(SYSTEM_PROMPT, context)
        )
    except Exception as exc:  # pragma: no cover - network
        return LLMResult(
            text=baseline,
            provider=provider,
            model=SETTINGS.llm.model,
            grounded=True,
            violations=[f"provider error, fell back to deterministic synthesis: {exc}"],
            fallback_used=True,
        )

    grounded, violations = validate_grounding(text, allowed)
    if not grounded:
        # An ungrounded answer is discarded outright, not patched up.
        return LLMResult(
            text=baseline,
            provider=provider,
            model=SETTINGS.llm.model,
            grounded=False,
            violations=violations + ["ungrounded answer rejected; deterministic text used"],
            fallback_used=True,
        )
    return LLMResult(
        text=text.strip(),
        provider=provider,
        model=SETTINGS.llm.model,
        grounded=True,
        violations=[],
    )


def provider_info() -> dict[str, Any]:
    return {
        "provider": SETTINGS.llm.provider,
        "model": SETTINGS.llm.model
        if SETTINGS.llm.provider != "deterministic"
        else "template-synthesis",
        "temperature": SETTINGS.llm.temperature,
    }
