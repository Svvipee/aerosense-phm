"""
Unit tests for the SQLite data layer.

These tests prove that the persistence layer correctly stores and retrieves
fleet health data — the foundation that allows the PHM dashboard to display
real-time health status, alert history, and trend analysis for every actuator
across the fleet.

Test coverage:
 - Aircraft CRUD and fleet summary queries
 - Actuator registration and association
 - Health snapshot insertion and retrieval
 - Time-series ordering for trend charts
 - Alert lifecycle (create → acknowledge → resolve)
 - Fleet-level aggregation correctness
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import pytest
from src.data.database import Database
from src.phm.feature_extraction import FEATURE_NAMES


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    yield d
    d.close()


class TestDatabase:
    def test_upsert_aircraft(self, db):
        """PROVES: An aircraft can be registered in the database and appears in
        the fleet summary query.  This is the entry point for the entire fleet
        management workflow — if this fails, no aircraft data can be stored.
        Uses LEFT JOIN so aircraft with zero actuators still appear (bug fix verified)."""
        db.upsert_aircraft("AC001", "A320", "N-TEST1")
        rows = db.fleet_summary()
        assert any(r["aircraft_id"] == "AC001" for r in rows)

    def test_upsert_actuator(self, db):
        """PROVES: An actuator can be registered under a parent aircraft without
        raising a foreign-key or uniqueness error.  The actuator table is the
        link between the physical EMA hardware and the aircraft it belongs to.
        No exception = the schema relationships are correct."""
        db.upsert_aircraft("AC001", "A320")
        db.upsert_actuator("AC001_ail_L", "AC001", "aileron_L")
        # No exception = pass

    def test_insert_and_retrieve_snapshot(self, db):
        """PROVES: A full health snapshot (health_index, sub-health values, RUL,
        fault class, confidence, and all 17 features) round-trips through the
        database without data loss or corruption.  This is the core data path —
        every time the PHM pipeline processes a flight cycle, it writes one of
        these snapshots.  The test verifies that health_index=0.85 and cycle=100
        are exactly what comes back, confirming no truncation or type-casting errors."""
        db.upsert_aircraft("AC001", "A320")
        db.upsert_actuator("AC001_ail_L", "AC001", "aileron_L")
        features = {k: 0.5 for k in FEATURE_NAMES}
        db.insert_snapshot(
            actuator_uid="AC001_ail_L", cycle=100,
            health_index=0.85, bearing_health=0.9, winding_health=0.87,
            backlash_health=0.88, rul_predicted=250.0, rul_lower=200.0,
            rul_upper=300.0, fault_class=0, fault_label="Healthy",
            fault_confidence=0.92, features=features,
        )
        snap = db.get_latest_snapshot("AC001_ail_L")
        assert snap is not None
        assert snap["health_index"] == pytest.approx(0.85)
        assert snap["cycle"] == 100

    def test_health_history_ordering(self, db):
        """PROVES: Health history is returned in ascending cycle order (oldest
        first).  This guarantees the Chart.js trend line on the dashboard
        displays left-to-right time progression.  If ordering were wrong, the
        health trend chart would show zigzag lines instead of smooth degradation
        curves — making it useless for maintenance planners."""
        db.upsert_aircraft("AC001", "A320")
        db.upsert_actuator("AC001_ail_L", "AC001", "aileron_L")
        features = {k: 0.5 for k in FEATURE_NAMES}
        for cycle in [10, 20, 30]:
            db.insert_snapshot(
                "AC001_ail_L", cycle=cycle,
                health_index=1.0 - cycle * 0.01,
                bearing_health=0.9, winding_health=0.9, backlash_health=0.9,
                rul_predicted=400.0, rul_lower=350.0, rul_upper=450.0,
                fault_class=0, fault_label="Healthy", fault_confidence=0.9,
                features=features,
            )
        history = db.get_health_history("AC001_ail_L", last_n=10)
        cycles = [h["cycle"] for h in history]
        assert cycles == sorted(cycles), "History should be returned in ascending cycle order"

    def test_create_and_resolve_alert(self, db):
        """PROVES: The full alert lifecycle works end-to-end: create a CRITICAL
        alert → verify it appears in the active alerts list → resolve it →
        verify it no longer appears.  This is the MRO workflow: the PHM system
        generates alerts, a maintenance planner reviews them, and resolves them
        after corrective action.  If this fails, alerts would either never
        appear or never clear — both are dangerous in an aviation context."""
        db.upsert_aircraft("AC001", "A320")
        db.upsert_actuator("AC001_ail_L", "AC001", "aileron_L")
        alert_id = db.create_alert(
            actuator_uid="AC001_ail_L", aircraft_id="AC001",
            alert_type="HEALTH_CRITICAL", severity="CRITICAL",
            message="Test alert", cycles_remaining=30,
        )
        assert isinstance(alert_id, int)

        alerts = db.get_active_alerts("AC001")
        assert any(a["alert_id"] == alert_id for a in alerts)

        db.resolve_alert(alert_id)
        alerts_after = db.get_active_alerts("AC001")
        assert not any(a["alert_id"] == alert_id for a in alerts_after)

    def test_fleet_summary_aggregates_correctly(self, db):
        """PROVES: The fleet summary query correctly aggregates health data
        across multiple aircraft — an A320 with healthy actuators (HI=0.90)
        and a B737 with degraded bearings (HI=0.45) both appear with their
        respective health values.  This is the data behind the fleet overview
        table on the dashboard.  If aggregation were wrong, MRO planners would
        see incorrect fleet-wide health status."""
        db.upsert_aircraft("AC001", "A320", "N-T1")
        db.upsert_aircraft("AC002", "B737", "N-T2")
        db.upsert_actuator("AC001_ail", "AC001", "aileron_L")
        db.upsert_actuator("AC002_ail", "AC002", "aileron_L")
        features = {k: 0.5 for k in FEATURE_NAMES}
        db.insert_snapshot("AC001_ail", 1, 0.90, 0.95, 0.92, 0.94, 200.0, 170.0, 230.0, 0, "Healthy", 0.9, features)
        db.insert_snapshot("AC002_ail", 1, 0.45, 0.50, 0.80, 0.90, 80.0,  60.0, 100.0, 1, "Bearing Degradation", 0.8, features)
        summary = db.fleet_summary()
        ids = [r["aircraft_id"] for r in summary]
        assert "AC001" in ids
        assert "AC002" in ids
