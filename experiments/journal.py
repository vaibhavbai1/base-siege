"""Checksummed, transactional progress for deterministic experiment units."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3

from experiments.storage import json_bytes, run_lock, sync_directory


class Journal:
    def __init__(self, connection):
        self.connection = connection

    def get(self, key):
        row = self.connection.execute("SELECT value, sha256 FROM items WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        value, digest = row
        if hashlib.sha256(value.encode()).hexdigest() != digest:
            raise ValueError(f"Progress checksum mismatch: {key}")
        return json.loads(value)

    def put(self, key, value):
        payload = json_bytes(value).decode()
        existing = self.get(key)
        if existing is not None:
            if json_bytes(existing).decode() != payload:
                raise ValueError(f"Cannot replace a completed experiment unit: {key}")
            return
        with self.connection:
            self.connection.execute("INSERT INTO items VALUES (?, ?, ?)",
                (key, payload, hashlib.sha256(payload.encode()).hexdigest()))

    def rows(self):
        for (key,) in self.connection.execute("SELECT key FROM items ORDER BY key"):
            yield key, self.get(key)

    def count(self):
        return self.connection.execute("SELECT COUNT(*) FROM items").fetchone()[0]


@contextmanager
def open_journal(path, contract, *, resume=False):
    """One writer, one immutable contract, one durable transaction per unit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with run_lock(path.parent, f".{path.name}.lock"):
        exists = path.exists()
        if exists and not resume:
            raise FileExistsError(f"Progress exists; use --resume with the same settings: {path}")
        if resume and not exists:
            raise FileNotFoundError(f"No progress to resume: {path}")
        connection = sqlite3.connect(path, timeout=0)
        try:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA journal_mode=DELETE")
            if not exists:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                connection.execute("CREATE TABLE items (key TEXT PRIMARY KEY, value TEXT NOT NULL, sha256 TEXT NOT NULL)")
                connection.execute("INSERT INTO metadata VALUES ('contract', ?)", (json_bytes(contract).decode(),))
                connection.commit()
                sync_directory(path.parent)
            saved = connection.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()
            if not saved or saved[0] != json_bytes(contract).decode():
                raise ValueError("Progress contract changed; restore the original inputs, code and settings")
            if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("Damaged progress database; restore a verified backup")
            yield Journal(connection)
        finally:
            connection.close()
