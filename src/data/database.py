"""
Persistent Data Store
======================

SQLite-backed storage for:
  - Fleet registry (aircraft and EMA metadata)
  - Time-series health snapshots (feature vectors per cycle)
  - Maintenance alerts and their resolution status

SQLite was chosen for the MVP because it requires zero infrastructure,
is embeddable in the demo, and is ACID-compliant for single-process use.
A production deployment would replace this with a time-series database
(e.g., InfluxDB, TimescaleDB) and a relational store (PostgreSQL) for
alert management, consistent with the architecture described in:
  Lee, J. et al. (2014). "Service Innovation and Smart Analytics for
  Industry 4.0 and Big Data Environment." Procedia CIRP, 16, 3–8.
"""

from __future__ import annotations

import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DB_PATH = Path("data/ema_phm.db")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    """Thread-safe SQLite wrapper for the EMA PHM system."""

    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # ------------------------------------------------------------------ #
    # Schema                                                              #
    # ------------------------------------------------------------------ #

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript("""
        PRAGMA journal_mode=WAL;

        CREATE TABLE IF NOT EXISTS aircraft (
            aircraft_id   TEXT PRIMARY KEY,
            ac_type       TEXT NOT NULL,
            tail_number   TEXT,
            n_cycles_total INTEGER DEFAULT 0,
            created_at    TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS actuators (
            actuator_uid  TEXT PRIMARY KEY,
            aircraft_id   TEXT NOT NULL,
            actuator_type TEXT NOT NULL,
            part_number   TEXT,
            serial_number TEXT,
            install_cycle INTEGER DEFAULT 0,
            created_at    TEXT NOT NULL,
            FOREIGN KEY(aircraft_id) REFERENCES aircraft(aircraft_id)
        );

        CREATE TABLE IF NOT EXISTS health_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            actuator_uid    TEXT NOT NULL,
            cycle           INTEGER NOT NULL,
            timestamp       TEXT NOT NULL,
            health_index    REAL,
            bearing_health  REAL,
            winding_health  REAL,
            backlash_health REAL,
            rul_predicted   REAL,
            rul_lower       REAL,
            rul_upper       REAL,
            fault_class     INTEGER,
            fault_label     TEXT,
            fault_confidence REAL,
            features_json   TEXT,
            FOREIGN KEY(actuator_uid) REFERENCES actuators(actuator_uid)
        );

        CREATE INDEX IF NOT EXISTS idx_snapshots_uid_cycle
            ON health_snapshots(actuator_uid, cycle);

        CREATE TABLE IF NOT EXISTS alerts (
            alert_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            actuator_uid    TEXT NOT NULL,
            aircraft_id     TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            alert_type      TEXT NOT NULL,
            severity        TEXT NOT NULL,
            message         TEXT NOT NULL,
            predicted_eol_cycle INTEGER,
            cycles_remaining    INTEGER,
            resolved        INTEGER DEFAULT 0,
            resolved_at     TEXT,
            acknowledged    INTEGER DEFAULT 0,
            ack_at          TEXT,
            FOREIGN KEY(actuator_uid) REFERENCES actuators(actuator_uid)
        );

        CREATE INDEX IF NOT EXISTS idx_alerts_aircraft
            ON alerts(aircraft_id, resolved);
        """)
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # Fleet registry                                                      #
    # ------------------------------------------------------------------ #

    def upsert_aircraft(
        self, aircraft_id: str, ac_type: str, tail_number: str = ""
    ) -> None:
        self._conn.execute("""
            INSERT INTO aircraft(aircraft_id, ac_type, tail_number, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(aircraft_id) DO UPDATE SET
                ac_type=excluded.ac_type,
                tail_number=COALESCE(excluded.tail_number, tail_number)
        """, (aircraft_id, ac_type, tail_number, _now()))
        self._conn.commit()

    def upsert_actuator(
        self,
        actuator_uid: str,
        aircraft_id: str,
        actuator_type: str,
        part_number: str = "",
        serial_number: str = "",
    ) -> None:
        self._conn.execute("""
            INSERT INTO actuators(actuator_uid, aircraft_id, actuator_type,
                                  part_number, serial_number, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(actuator_uid) DO NOTHING
        """, (actuator_uid, aircraft_id, actuator_type, part_number, serial_number, _now()))
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # Health snapshots                                                    #
    # ------------------------------------------------------------------ #

    def insert_snapshot(
        self,
        actuator_uid: str,
        cycle: int,
        health_index: float,
        bearing_health: float,
        winding_health: float,
        backlash_health: float,
        rul_predicted: float,
        rul_lower: float,
        rul_upper: float,
        fault_class: int,
        fault_label: str,
        fault_confidence: float,
        features: dict[str, float],
    ) -> None:
        self._conn.execute("""
            INSERT INTO health_snapshots(
                actuator_uid, cycle, timestamp,
                health_index, bearing_health, winding_health, backlash_health,
                rul_predicted, rul_lower, rul_upper,
                fault_class, fault_label, fault_confidence, features_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            actuator_uid, cycle, _now(),
            health_index, bearing_health, winding_health, backlash_health,
            rul_predicted, rul_lower, rul_upper,
            fault_class, fault_label, fault_confidence,
            json.dumps(features),
        ))
        self._conn.commit()

    def get_health_history(
        self, actuator_uid: str, last_n: int = 200
    ) -> list[dict]:
        cur = self._conn.execute("""
            SELECT * FROM health_snapshots
            WHERE actuator_uid = ?
            ORDER BY cycle DESC
            LIMIT ?
        """, (actuator_uid, last_n))
        rows = cur.fetchall()
        return [dict(r) for r in reversed(rows)]

    def get_latest_snapshot(self, actuator_uid: str) -> dict | None:
        cur = self._conn.execute("""
            SELECT * FROM health_snapshots
            WHERE actuator_uid = ?
            ORDER BY cycle DESC
            LIMIT 1
        """, (actuator_uid,))
        row = cur.fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------ #
    # Alert management                                                    #
    # ------------------------------------------------------------------ #

    def create_alert(
        self,
        actuator_uid: str,
        aircraft_id: str,
        alert_type: str,
        severity: str,
        message: str,
        predicted_eol_cycle: int | None = None,
        cycles_remaining: int | None = None,
    ) -> int:
        cur = self._conn.execute("""
            INSERT INTO alerts(
                actuator_uid, aircraft_id, created_at,
                alert_type, severity, message,
                predicted_eol_cycle, cycles_remaining
            ) VALUES (?,?,?,?,?,?,?,?)
        """, (
            actuator_uid, aircraft_id, _now(),
            alert_type, severity, message,
            predicted_eol_cycle, cycles_remaining,
        ))
        self._conn.commit()
        return cur.lastrowid

    def get_active_alerts(self, aircraft_id: str | None = None) -> list[dict]:
        if aircraft_id:
            cur = self._conn.execute("""
                SELECT a.*, act.actuator_type FROM alerts a
                JOIN actuators act ON a.actuator_uid = act.actuator_uid
                WHERE a.aircraft_id = ? AND a.resolved = 0
                ORDER BY a.created_at DESC
            """, (aircraft_id,))
        else:
            cur = self._conn.execute("""
                SELECT a.*, act.actuator_type FROM alerts a
                JOIN actuators act ON a.actuator_uid = act.actuator_uid
                WHERE a.resolved = 0
                ORDER BY a.created_at DESC
            """)
        return [dict(r) for r in cur.fetchall()]

    def acknowledge_alert(self, alert_id: int) -> None:
        self._conn.execute(
            "UPDATE alerts SET acknowledged=1, ack_at=? WHERE alert_id=?",
            (_now(), alert_id)
        )
        self._conn.commit()

    def resolve_alert(self, alert_id: int) -> None:
        self._conn.execute(
            "UPDATE alerts SET resolved=1, resolved_at=? WHERE alert_id=?",
            (_now(), alert_id)
        )
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # Fleet overview queries                                              #
    # ------------------------------------------------------------------ #

    def fleet_summary(self) -> list[dict]:
        """Return one row per aircraft with latest health metrics."""
        cur = self._conn.execute("""
            SELECT
                ac.aircraft_id,
                ac.ac_type,
                ac.tail_number,
                COUNT(DISTINCT act.actuator_uid) AS n_actuators,
                MIN(hs.health_index) AS fleet_min_health,
                AVG(hs.health_index) AS fleet_avg_health,
                MIN(hs.rul_predicted) AS min_rul,
                SUM(CASE WHEN al.resolved=0 THEN 1 ELSE 0 END) AS open_alerts
            FROM aircraft ac
            LEFT JOIN actuators act ON ac.aircraft_id = act.aircraft_id
            LEFT JOIN health_snapshots hs ON act.actuator_uid = hs.actuator_uid
                AND hs.id = (
                    SELECT MAX(id) FROM health_snapshots
                    WHERE actuator_uid = act.actuator_uid
                )
            LEFT JOIN alerts al ON ac.aircraft_id = al.aircraft_id AND al.resolved=0
            GROUP BY ac.aircraft_id
            ORDER BY fleet_min_health ASC
        """)
        return [dict(r) for r in cur.fetchall()]

    def actuator_summary(self, aircraft_id: str) -> list[dict]:
        """Return latest health snapshot for every actuator on an aircraft."""
        cur = self._conn.execute("""
            SELECT
                act.actuator_uid,
                act.actuator_type,
                hs.cycle,
                hs.health_index,
                hs.bearing_health,
                hs.winding_health,
                hs.backlash_health,
                hs.rul_predicted,
                hs.rul_lower,
                hs.rul_upper,
                hs.fault_label,
                hs.fault_confidence,
                hs.timestamp
            FROM actuators act
            LEFT JOIN health_snapshots hs ON act.actuator_uid = hs.actuator_uid
                AND hs.id = (
                    SELECT MAX(id) FROM health_snapshots
                    WHERE actuator_uid = act.actuator_uid
                )
            WHERE act.aircraft_id = ?
            ORDER BY hs.health_index ASC
        """, (aircraft_id,))
        return [dict(r) for r in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()
