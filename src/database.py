"""Small SQLite persistence layer for the hackathon prototype."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS patients (
            id TEXT PRIMARY KEY,
            local_patient_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            sex TEXT NOT NULL,
            phone TEXT,
            village TEXT,
            abha_number TEXT,
            diabetes_duration INTEGER,
            consent INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS screenings (
            id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL REFERENCES patients(id),
            right_image TEXT,
            left_image TEXT,
            right_quality TEXT NOT NULL,
            left_quality TEXT NOT NULL,
            right_result TEXT NOT NULL,
            left_result TEXT NOT NULL,
            history TEXT NOT NULL,
            priority TEXT NOT NULL,
            priority_rank INTEGER NOT NULL,
            triage_reasons TEXT NOT NULL,
            rule_version TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS reviews (
            id TEXT PRIMARY KEY,
            screening_id TEXT UNIQUE NOT NULL REFERENCES screenings(id),
            doctor_name TEXT NOT NULL,
            doctor_registration TEXT NOT NULL,
            right_grade INTEGER,
            left_grade INTEGER,
            action TEXT NOT NULL,
            notes TEXT,
            prescription TEXT,
            follow_up_date TEXT,
            reviewed_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            action TEXT NOT NULL,
            actor TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
        with self.connect() as connection:
            connection.executescript(schema)

    def create_patient(self, data: dict) -> str:
        patient_id = str(uuid.uuid4())
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO patients
                (id, local_patient_id, name, age, sex, phone, village, abha_number,
                 diabetes_duration, consent, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    patient_id,
                    data["local_patient_id"],
                    data["name"],
                    data["age"],
                    data["sex"],
                    data.get("phone", ""),
                    data.get("village", ""),
                    data.get("abha_number", ""),
                    data.get("diabetes_duration", 0),
                    int(data.get("consent", False)),
                    utc_now(),
                ),
            )
            self._audit(connection, "patient", patient_id, "created", "village-worker")
        return patient_id

    def create_screening(self, data: dict) -> str:
        screening_id = str(uuid.uuid4())
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO screenings
                (id, patient_id, right_image, left_image, right_quality, left_quality,
                 right_result, left_result, history, priority, priority_rank,
                 triage_reasons, rule_version, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    screening_id,
                    data["patient_id"],
                    data.get("right_image"),
                    data.get("left_image"),
                    json.dumps(data["right_quality"]),
                    json.dumps(data["left_quality"]),
                    json.dumps(data["right_result"]),
                    json.dumps(data["left_result"]),
                    json.dumps(data["history"]),
                    data["triage"]["priority"],
                    data["triage"]["priority_rank"],
                    json.dumps(data["triage"]["reasons"]),
                    data["triage"]["rule_version"],
                    data.get("status", "Awaiting doctor review"),
                    utc_now(),
                ),
            )
            self._audit(connection, "screening", screening_id, "submitted", "village-worker")
        return screening_id

    def save_review(self, screening_id: str, data: dict) -> str:
        review_id = str(uuid.uuid4())
        with self.connect() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO reviews
                (id, screening_id, doctor_name, doctor_registration, right_grade,
                 left_grade, action, notes, prescription, follow_up_date, reviewed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    review_id,
                    screening_id,
                    data["doctor_name"],
                    data["doctor_registration"],
                    data.get("right_grade"),
                    data.get("left_grade"),
                    data["action"],
                    data.get("notes", ""),
                    data.get("prescription", ""),
                    data.get("follow_up_date", ""),
                    utc_now(),
                ),
            )
            status = "Recapture requested" if data["action"] == "Request image recapture" else "Doctor reviewed"
            connection.execute("UPDATE screenings SET status = ? WHERE id = ?", (status, screening_id))
            self._audit(connection, "screening", screening_id, "reviewed", data["doctor_name"])
        return review_id

    def list_patients(self) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM patients ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

    def list_screenings(self) -> list[dict]:
        query = """
        SELECT s.*, p.name AS patient_name, p.local_patient_id, p.age, p.sex,
               p.phone, p.village, p.abha_number, p.diabetes_duration,
               r.id AS review_id, r.doctor_name, r.doctor_registration,
               r.right_grade AS reviewed_right_grade, r.left_grade AS reviewed_left_grade,
               r.action AS review_action, r.notes AS review_notes,
               r.prescription, r.follow_up_date, r.reviewed_at
        FROM screenings s
        JOIN patients p ON p.id = s.patient_id
        LEFT JOIN reviews r ON r.screening_id = s.id
        ORDER BY s.priority_rank ASC, s.created_at DESC
        """
        with self.connect() as connection:
            rows = connection.execute(query).fetchall()
        return [self._decode_screening(dict(row)) for row in rows]

    def get_screening(self, screening_id: str) -> dict | None:
        return next((item for item in self.list_screenings() if item["id"] == screening_id), None)

    def counts(self) -> dict:
        screenings = self.list_screenings()
        return {
            "patients": len(self.list_patients()),
            "awaiting": sum(item["status"] == "Awaiting doctor review" for item in screenings),
            "urgent": sum(item["priority"] == "URGENT" and item["status"] != "Doctor reviewed" for item in screenings),
            "reviewed": sum(item["status"] == "Doctor reviewed" for item in screenings),
            "recapture": sum(item["status"] == "Recapture requested" for item in screenings),
        }

    @staticmethod
    def _decode_screening(item: dict) -> dict:
        for key in ("right_quality", "left_quality", "right_result", "left_result", "history", "triage_reasons"):
            item[key] = json.loads(item[key])
        return item

    @staticmethod
    def _audit(connection: sqlite3.Connection, entity_type: str, entity_id: str, action: str, actor: str) -> None:
        connection.execute(
            "INSERT INTO audit_events (entity_type, entity_id, action, actor, created_at) VALUES (?, ?, ?, ?, ?)",
            (entity_type, entity_id, action, actor, utc_now()),
        )

