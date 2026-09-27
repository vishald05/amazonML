"""Memory-bounded candidate generation backed by SQLite.

The challenge sources are too large for a multiprocessing Python inverted index.
This module keeps the blocking keys and records on disk and returns a deterministic
union of cheap, high-signal blocks.
"""

import csv
import re
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, Set

from normalization import (
    normalize_address,
    normalize_name,
    normalize_transliterate,
)


def _keys(name: str, address: str):
    name = normalize_name(name)
    address = normalize_address(address)
    tokens = sorted({token for token in name.split() if len(token) >= 3})
    yield "name", name
    ascii_name = normalize_transliterate(name)
    if ascii_name and ascii_name != name:
        yield "name_ascii", ascii_name
    if address:
        yield "address", address
    if tokens:
        yield "token", tokens[0]
        if len(tokens) > 1:
            yield "token2", " ".join(tokens[:2])


class SQLiteCandidateGenerator:
    """Build and query a compact on-disk blocking index."""

    def __init__(self, db_path: Path, max_candidates: int = 200):
        self.db_path = Path(db_path)
        self.max_candidates = max_candidates
        self.connection = None

    def build(self, source_paths: Iterable[Path]) -> None:
        if self.db_path.exists():
            connection = sqlite3.connect(self.db_path)
            ready = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='index' AND name='blocks_lookup'"
            ).fetchone()
            connection.close()
            if ready:
                self.connection = sqlite3.connect(self.db_path)
                return
        if self.db_path.exists():
            self.db_path.unlink()
        self.connection = sqlite3.connect(self.db_path)
        self.connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=MEMORY;
            CREATE TABLE records (
                entity_id TEXT PRIMARY KEY,
                business_name TEXT NOT NULL,
                business_address TEXT NOT NULL,
                country TEXT NOT NULL
            );
            CREATE TABLE blocks (kind TEXT NOT NULL, block_key TEXT NOT NULL,
                                 entity_id TEXT NOT NULL);
            """
        )
        record_sql = "INSERT INTO records VALUES (?, ?, ?, ?)"
        block_sql = "INSERT INTO blocks VALUES (?, ?, ?)"
        records, blocks = [], []
        for path in source_paths:
            with open(path, encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                for row in reader:
                    entity_id = row["entity_id"]
                    name = row.get("business_name") or ""
                    address = row.get("business_address") or ""
                    country = row.get("country") or ""
                    records.append((entity_id, name, address, country))
                    for kind, key in _keys(name, address):
                        if key:
                            blocks.append((kind, key, entity_id))
                    if len(records) >= 10000:
                        self.connection.executemany(record_sql, records)
                        self.connection.executemany(block_sql, blocks)
                        self.connection.commit()
                        records.clear()
                        blocks.clear()
        if records:
            self.connection.executemany(record_sql, records)
            self.connection.executemany(block_sql, blocks)
            self.connection.commit()
        self.connection.executescript(
            "CREATE INDEX blocks_lookup ON blocks(kind, block_key);"
        )
        self.connection.commit()

    def candidates_for(self, name: str, address: str, country: str) -> Set[str]:
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path)
        keys = list(_keys(name, address))
        scores: Dict[str, int] = {}
        for kind, key in keys:
            rows = self.connection.execute(
                "SELECT entity_id FROM blocks WHERE kind=? AND block_key=?",
                (kind, key),
            )
            for (entity_id,) in rows:
                scores[entity_id] = scores.get(entity_id, 0) + 1
        ranked = sorted(scores, key=lambda entity_id: (-scores[entity_id], entity_id))
        return set(ranked[: self.max_candidates])

    def records_for(self, entity_ids: Iterable[str]) -> Dict[str, dict]:
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path)
        ids = list(entity_ids)
        if not ids:
            return {}
        result = {}
        for start in range(0, len(ids), 500):
            batch = ids[start:start + 500]
            marks = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                f"SELECT entity_id,business_name,business_address,country "
                f"FROM records WHERE entity_id IN ({marks})", batch
            )
            for entity_id, name, address, country in rows:
                result[entity_id] = {
                    "business_name": name,
                    "business_address": address,
                    "country": country,
                }
        return result

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None