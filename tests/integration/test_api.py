"""Phase 12 gate: every endpoint works and the contract holds."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.main import app


@pytest.fixture(scope="module")
def client():  # noqa: ANN201
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["documents"] > 100
    assert body["coverage_entries"] > 50


def test_query_observed_absence(client):
    body = client.post(
        "/query",
        json={"question": "Were there any surface contacts in Sector Alpha between 04:00 and 04:20?"},
    ).json()
    assert body["state"] == "OBSERVED_ABSENCE"
    assert body["coverage"]["coverage_fraction"] > 0.9
    assert body["confidence"] > 0.6
    assert body["evidence"]
    assert body["gaps"], "the 4-minute Grid B7 gap must still be reported"


def test_query_unknown_on_blind_window(client):
    body = client.post(
        "/query",
        json={"question": "Were there any contacts in Sector Alpha between 04:07 and 04:11?"},
    ).json()
    assert body["state"] == "UNKNOWN"
    assert body["confidence"] <= 0.45
    assert "no contacts were observed" not in body["answer"].lower()


def test_query_contradiction(client):
    body = client.post("/query", json={"question": "What vessel was detected near Grid B7?"}).json()
    assert body["state"] == "CONTRADICTION"
    assert body["contradictions"]
    values = {str(c["value"]) for c in body["contradictions"][0]["claims"]}
    assert {"V-17", "V-21"} <= values


def test_query_trace_has_every_stage(client):
    body = client.post(
        "/query",
        json={"question": "What contacts were observed in Grid B1 between 05:00 and 05:10?"},
    ).json()
    stages = {s["name"] for s in body["trace"]["stages"]}
    for expected in (
        "decomposition",
        "dense_sparse_retrieval",
        "coverage_check",
        "fusion_rerank",
        "evidence_classification",
        "contradiction_detection",
        "confidence_calculation",
        "llm_synthesis",
    ):
        assert expected in stages
    assert body["trace"]["total_latency_ms"] > 0


def test_coverage_endpoint_is_independent_of_retrieval(client):
    body = client.get(
        "/coverage", params={"region": "grid_b7", "start": "04:07", "end": "04:11"}
    ).json()
    assert body["status"] == "NOT_OBSERVED"
    assert body["covered_fraction"] == 0.0
    assert body["absence_claim_supported"] is False


def test_coverage_endpoint_rejects_bad_input(client):
    assert client.get(
        "/coverage", params={"region": "grid_b7", "start": "nonsense", "end": "04:11"}
    ).status_code == 400


def test_evidence_endpoint(client):
    body = client.get("/evidence/RADAR-221").json()
    assert body["record"]["track_id"] == "T-88"
    assert "coverage_at_record" in body
    assert body["sensor"]["reliability"] > 0.5
    assert client.get("/evidence/NOPE-999").status_code == 404


def test_metrics_endpoint(client):
    client.post("/query", json={"question": "Were there any contacts in Grid A1 between 04:00 and 04:20?"})
    body = client.get("/metrics").json()
    assert body["queries_total"] >= 1
    assert body["index"]["documents"] > 100
    assert body["latency"]["mean_ms"] > 0


def test_ingest_records_and_coverage(client):
    before = client.get("/health").json()
    payload = {
        "records": [
            {
                "record_id": "TEST-INGEST-1",
                "modality": "surface_radar",
                "sensor": "radar_01",
                "timestamp": "2026-08-22T05:50:00+00:00",
                "region": "grid_a1",
                "text": "Surface track T-99 held in Grid A1: heading 010 degrees, speed 12 knots.",
                "reliability": 0.9,
                "track_id": "T-99",
            }
        ],
        "coverage_entries": [
            {
                "entry_id": "TEST-COV-1",
                "region": "grid_a1",
                "time_start": "2026-08-22T05:45:00+00:00",
                "time_end": "2026-08-22T05:55:00+00:00",
                "modality": "surface_radar",
                "sensor": "radar_01",
                "coverage_status": "OBSERVED",
                "coverage_confidence": 0.95,
            }
        ],
        "rebuild_index": False,
    }
    body = client.post("/ingest", json=payload).json()
    assert body["records_added"] == 1
    assert body["coverage_entries_added"] == 1
    after = client.get("/health").json()
    assert after["documents"] == before["documents"] + 1
    assert after["coverage_entries"] == before["coverage_entries"] + 1


def test_evaluation_endpoint_runs_retrieval_suite(client):
    body = client.post("/evaluation/run", json={"suite": "retrieval"}).json()
    assert body["retrieval_only"]["questions"] == 50
    assert body["environment"]["embedding_model"]
    assert body["environment"]["dataset_version"]
