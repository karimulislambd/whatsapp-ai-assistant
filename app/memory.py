"""SQLite persistence: conversation memory, user mode, webhook dedupe and rate limiting.

User identifiers (WhatsApp ``wa_id`` phone numbers, web-demo session ids) are never stored
raw — every table is keyed by a SHA-256 hash of the identifier.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_hash  TEXT    NOT NULL,
    mode       TEXT    NOT NULL,
    role       TEXT    NOT NULL CHECK (role IN ('user', 'assistant')),
    content    TEXT    NOT NULL,
    created_at REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_user ON messages (user_hash, mode, id);

CREATE TABLE IF NOT EXISTS users (
    user_hash  TEXT PRIMARY KEY,
    mode       TEXT NOT NULL DEFAULT 'chat',
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS processed_messages (
    message_id TEXT PRIMARY KEY,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS rate_events (
    user_hash TEXT NOT NULL,
    ts        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rate_user ON rate_events (user_hash, ts);
"""

DEFAULT_MODE = "chat"
DEDUPE_TTL_SECONDS = 7 * 24 * 3600  # Meta retries for up to ~7 days


def hash_user_id(raw_id: str) -> str:
    """One-way SHA-256 hash of a user identifier (phone number / session id)."""
    return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()


class Store:
    """Thin, thread-safe wrapper around a single SQLite connection.

    Every query is a sub-millisecond indexed lookup, so calling it from async code is fine;
    WAL mode keeps readers and the single writer from blocking each other.
    """

    def __init__(self, db_path: str, clock: Callable[[], float] = time.time) -> None:
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._clock = clock
        with self._lock:
            if db_path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ memory
    def add_message(self, user_hash: str, mode: str, role: str, content: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages (user_hash, mode, role, content, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_hash, mode, role, content, self._clock()),
            )

    def add_turn(
        self, user_hash: str, mode: str, user_msg: str, reply: str, max_turns: int
    ) -> None:
        """Store a user/assistant exchange and trim history to the last ``max_turns`` turns."""
        self.add_message(user_hash, mode, "user", user_msg)
        self.add_message(user_hash, mode, "assistant", reply)
        self.trim(user_hash, mode, max_turns)

    def trim(self, user_hash: str, mode: str, max_turns: int) -> None:
        keep = max_turns * 2  # one turn = user message + assistant reply
        with self._lock:
            self._conn.execute(
                "DELETE FROM messages WHERE user_hash = ? AND mode = ? AND id NOT IN ("
                "  SELECT id FROM messages WHERE user_hash = ? AND mode = ? "
                "  ORDER BY id DESC LIMIT ?)",
                (user_hash, mode, user_hash, mode, keep),
            )

    def get_history(self, user_hash: str, mode: str, max_turns: int) -> list[dict[str, str]]:
        """Return the last ``max_turns`` turns as OpenAI-style ``{"role", "content"}`` dicts."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM messages WHERE user_hash = ? AND mode = ? "
                "ORDER BY id DESC LIMIT ?",
                (user_hash, mode, max_turns * 2),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def clear_history(self, user_hash: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM messages WHERE user_hash = ?", (user_hash,))

    # -------------------------------------------------------------------- mode
    def get_mode(self, user_hash: str) -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT mode FROM users WHERE user_hash = ?", (user_hash,)
            ).fetchone()
        return row["mode"] if row else DEFAULT_MODE

    def set_mode(self, user_hash: str, mode: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO users (user_hash, mode, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(user_hash) DO UPDATE SET mode = excluded.mode, "
                "updated_at = excluded.updated_at",
                (user_hash, mode, self._clock()),
            )

    # ------------------------------------------------------------- idempotency
    def mark_processed(self, message_id: str) -> bool:
        """Atomically record a webhook message id. Returns False if it was already seen."""
        now = self._clock()
        with self._lock:
            self._conn.execute(
                "DELETE FROM processed_messages WHERE created_at < ?", (now - DEDUPE_TTL_SECONDS,)
            )
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO processed_messages (message_id, created_at) VALUES (?, ?)",
                (message_id, now),
            )
        return cur.rowcount == 1

    # ------------------------------------------------------------ rate limiting
    def allow_request(self, user_hash: str, limit: int, window_seconds: int) -> bool:
        """Sliding-window rate limiter: allow at most ``limit`` events per ``window_seconds``."""
        now = self._clock()
        with self._lock:
            self._conn.execute("DELETE FROM rate_events WHERE ts <= ?", (now - window_seconds,))
            (count,) = self._conn.execute(
                "SELECT COUNT(*) FROM rate_events WHERE user_hash = ?", (user_hash,)
            ).fetchone()
            if count >= limit:
                return False
            self._conn.execute(
                "INSERT INTO rate_events (user_hash, ts) VALUES (?, ?)", (user_hash, now)
            )
        return True
