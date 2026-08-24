"""Streamlit operator console.

Shows the answer, the coverage bar and timeline, the evidence, the contradictions, the
confidence breakdown and the full retrieval trace. The point of the UI is that an operator
can see *why* the system is or is not confident, and can tell "we looked and saw nothing"
apart from "we never looked" at a glance.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.config import SETTINGS  # noqa: E402
from app.coverage.ledger import get_ledger  # noqa: E402
from app.dataset import world  # noqa: E402
from app.models.schemas import AnswerState, TimeRange  # noqa: E402
from app.reasoning import llm  # noqa: E402
from app.reasoning.pipeline import answer_question  # noqa: E402
from app.retrieval.hybrid import get_retriever  # noqa: E402

st.set_page_config(page_title="Mission Intelligence", page_icon="🛰️", layout="wide")

STATE_STYLE = {
    "OBSERVED_ABSENCE": ("#1b5e20", "Observed absence — we looked and saw nothing"),
    "PRESENCE": ("#0d47a1", "Presence — contacts were detected"),
    "UNKNOWN": ("#e65100", "Unknown — insufficient observation coverage"),
    "CONTRADICTION": ("#b71c1c", "Contradiction — sources disagree"),
}

DEMOS = [
    "Were there any surface contacts in Sector Alpha between 04:00 and 04:20?",
    "Were there any contacts in Sector Alpha between 04:07 and 04:11?",
    "What vessel was detected near Grid B7?",
    "Is the vessel detected at 05:20 the same vessel tracked at 04:00?",
    "Can we conclude that no contacts were present in Sector C?",
    "Was vessel V-17 observed after 04:00?",
    "Were there any suspicious RF emissions in Sector Alpha during the last hour?",
]


@st.cache_resource(show_spinner="Loading index and coverage ledger…")
def _boot():  # noqa: ANN202
    return get_retriever(), get_ledger()


def coverage_bar(fraction: float, width: int = 30) -> str:
    filled = int(round(fraction * width))
    return "█" * filled + "░" * (width - filled)


def render_timeline(region: str, window: TimeRange, ledger, buckets: int = 24) -> None:  # noqa: ANN001
    """One row per atomic sub-region, so a blind pocket is visible rather than averaged."""
    grids = world.atomic_regions(region)
    minutes = max(1, int(window.duration_seconds // 60))
    buckets = max(4, min(buckets, minutes if minutes >= 4 else 4))
    step = window.duration_seconds / buckets

    header_ticks = "".join(
        (window.start + timedelta(seconds=step * i)).strftime("%M")[-1] for i in range(buckets)
    )
    lines = [
        f"{'':<14}{window.start.strftime('%H:%M')}{'':<{max(0, buckets - 10)}}"
        f"{window.end.strftime('%H:%M')}",
        f"{'minute':<14}{header_ticks}",
    ]
    for grid in grids:
        cells = []
        for i in range(buckets):
            slot = TimeRange(
                start=window.start + timedelta(seconds=step * i),
                end=window.start + timedelta(seconds=step * (i + 1)),
            )
            fraction = ledger.check(grid, slot).covered_fraction
            cells.append("█" if fraction >= 0.85 else "▓" if fraction >= 0.35 else "░")
        overall = ledger.check(grid, window)
        lines.append(f"{grid:<14}{''.join(cells)}  {overall.covered_fraction:>4.0%}")
    lines.append("")
    lines.append("█ observed   ▓ partial/degraded   ░ unobserved")
    st.code("\n".join(lines), language=None)


def main() -> None:
    retriever, ledger = _boot()

    st.title("🛰️ Mission Intelligence — Coverage-Aware Retrieval")
    st.caption(
        "An empty retrieval result is not evidence of absence. Observation coverage is "
        "modelled explicitly and evaluated independently of the retrieval index."
    )

    with st.sidebar:
        st.header("System")
        st.write(f"**Documents:** {retriever.build_info['documents']}")
        st.write(f"**Coverage entries:** {len(ledger.entries)}")
        st.write(f"**Embedding:** `{retriever.build_info['embedding_model']}`")
        st.write(f"**Dense backend:** `{retriever.build_info['dense_backend']}`")
        st.write(f"**LLM:** `{llm.provider_info()['provider']}` / `{llm.provider_info()['model']}`")
        st.write(f"**Dataset:** `{SETTINGS.dataset_version}`")
        st.divider()
        st.header("Coverage ledger explorer")
        region = st.selectbox(
            "Region", sorted(world.REGIONS), index=sorted(world.REGIONS).index("sector_alpha")
        )
        col_a, col_b = st.columns(2)
        start_h = col_a.slider("From (HH:MM)", 3.5, 6.0, 4.0, 0.05)
        end_h = col_b.slider("To (HH:MM)", 3.5, 6.0, 4.35, 0.05)
        if end_h > start_h:
            win = TimeRange(
                start=world.MISSION_START.replace(hour=3, minute=0) + timedelta(hours=start_h),
                end=world.MISSION_START.replace(hour=3, minute=0) + timedelta(hours=end_h),
            )
            report = ledger.check(region, win)
            st.write(f"`{report.human_summary()}`")
            st.write(
                f"absence claim supported: **{report.absence_claim_supported}**"
                + (f" — {report.absence_block_reason}" if report.absence_block_reason else "")
            )

    st.subheader("Operator question")
    choice = st.selectbox("Demo questions", ["(type your own)"] + DEMOS)
    default = "" if choice == "(type your own)" else choice
    question = st.text_input("Question", value=default, key=f"q-{choice}")

    if not st.button("Ask", type="primary") or not question.strip():
        st.info("Pick a demo question or type your own, then press **Ask**.")
        return

    with st.spinner("Retrieving, checking coverage, classifying evidence…"):
        answer = asyncio.run(answer_question(question))

    colour, blurb = STATE_STYLE.get(answer.state.value, ("#333", ""))
    st.markdown(
        f"<div style='background:{colour};color:white;padding:16px 20px;border-radius:10px'>"
        f"<div style='font-size:13px;opacity:.85;letter-spacing:.08em'>{blurb.upper()}</div>"
        f"<div style='font-size:21px;line-height:1.45;margin-top:8px'>{answer.answer}</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("State", answer.state.value)
    m2.metric("Confidence", f"{answer.confidence:.2f}")
    m3.metric("Coverage", f"{answer.coverage.coverage_fraction:.0%}")
    m4.metric("Latency", f"{answer.meta['total_latency_ms']:.0f} ms")

    st.subheader("Observation coverage")
    st.code(
        f"{answer.coverage.region}  {coverage_bar(answer.coverage.coverage_fraction)}  "
        f"{answer.coverage.coverage_fraction:.0%} observed   "
        f"[{answer.coverage.status.value}]   {answer.coverage.time_range}",
        language=None,
    )
    if answer.coverage.missing_modalities:
        st.warning("No coverage from: " + ", ".join(answer.coverage.missing_modalities))

    st.markdown("**Timeline**")
    plan_region = answer.coverage.region
    tr = answer.plan.time_range if answer.plan and answer.plan.time_range else TimeRange(
        start=world.MISSION_START, end=world.MISSION_NOW
    )
    render_timeline(plan_region, tr, ledger)

    if answer.gaps:
        st.subheader("Gaps")
        for gap in answer.gaps:
            st.markdown(f"- ⚠️ {gap}")

    if answer.contradictions:
        st.subheader("Contradictions")
        for c in answer.contradictions:
            with st.expander(
                f"[{c['severity_label'].upper()}] {c['dimension']} — {c['reason']}", expanded=True
            ):
                st.dataframe(
                    pd.DataFrame(c["claims"]),
                    hide_index=True,
                    use_container_width=True,
                )
                st.caption(
                    "The system does not resolve this. No majority vote, no highest-score wins."
                )
    related = answer.meta.get("related_contradictions") or []
    if related:
        with st.expander(f"Related disagreements outside the queried window ({len(related)})"):
            st.dataframe(pd.DataFrame(related), hide_index=True, use_container_width=True)

    st.subheader("Evidence")
    if answer.evidence:
        claim_ids = set(answer.meta.get("claim_evidence_ids", []))
        rows = [
            {
                "supports claim": "✅" if e["id"] in claim_ids else "",
                "id": e["id"],
                "state": e["state"],
                "source": e["source"],
                "sensor": e["sensor"],
                "region": e["region"],
                "time": e["time_range"],
                "reliability": e["reliability"],
                "claim": e["claim"],
            }
            for e in answer.evidence
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.write("No evidence retrieved.")

    if answer.meta.get("association"):
        st.subheader("Multi-hop association")
        assoc = answer.meta["association"]
        st.write(f"**Verdict:** `{assoc['verdict']}`")
        st.json(assoc, expanded=False)

    st.subheader("Confidence breakdown")
    if answer.confidence_breakdown:
        cb = answer.confidence_breakdown
        st.dataframe(
            pd.DataFrame(
                [
                    {"feature": "coverage", "value": cb.coverage},
                    {"feature": "source reliability", "value": cb.source_reliability},
                    {"feature": "recency", "value": cb.recency},
                    {"feature": "retrieval agreement", "value": cb.retrieval_agreement},
                    {"feature": "evidence support", "value": cb.evidence_support},
                    {"feature": "contradiction penalty", "value": -cb.contradiction_penalty},
                    {"feature": "stale penalty", "value": -cb.stale_penalty},
                    {"feature": "absence coverage multiplier", "value": cb.absence_coverage_multiplier},
                    {"feature": "FINAL", "value": cb.confidence},
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Deterministic. Computed in Python before the language model is called; the "
            "model cannot change it."
        )

    if answer.uncertainty:
        st.subheader("Uncertainty and exclusions")
        for note in answer.uncertainty:
            st.markdown(f"- {note}")

    with st.expander("Retrieval trace (why these records?)"):
        if answer.plan:
            st.markdown("**Query plan**")
            st.json(
                {
                    "intent": answer.plan.intent.value,
                    "region": answer.plan.region,
                    "time_range": answer.plan.time_range.label() if answer.plan.time_range else None,
                    "entities": answer.plan.entities,
                    "modalities_explicit": answer.plan.modalities_explicit,
                    "notes": answer.plan.notes,
                    "subqueries": [
                        {
                            "id": s.subquery_id,
                            "type": s.type.value,
                            "text": s.text,
                            "rationale": s.rationale,
                        }
                        for s in answer.plan.subqueries
                    ],
                },
                expanded=False,
            )
        if answer.trace:
            st.markdown("**Stage latencies**")
            st.dataframe(
                pd.DataFrame(
                    [{"stage": s.name, "ms": s.latency_ms} for s in answer.trace.stages]
                ),
                hide_index=True,
                use_container_width=True,
            )
            st.markdown("**Retrieved records**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "id": d.record_id,
                            "score": round(d.score, 4),
                            "dense_rank": d.dense_rank,
                            "bm25_rank": d.sparse_rank,
                            "rrf": round(d.fusion_score, 5),
                            "why": "; ".join(d.why),
                            "subquery": d.subquery_id,
                        }
                        for d in answer.trace.retrieved
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
