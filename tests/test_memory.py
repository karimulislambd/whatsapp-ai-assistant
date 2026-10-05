"""SQLite store: hashing, memory trimming, modes, dedupe and the rate limiter."""

import sqlite3

from app.memory import Store, hash_user_id


class FakeClock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_hash_is_sha256_and_stable():
    h = hash_user_id("wa:8801700000000")
    assert len(h) == 64 and h == hash_user_id("wa:8801700000000")
    assert h != hash_user_id("wa:8801700000001")


def test_memory_is_trimmed_to_last_n_turns(tmp_path):
    store = Store(str(tmp_path / "m.db"))
    user = hash_user_id("u")
    for i in range(15):
        store.add_turn(user, "chat", f"q{i}", f"a{i}", max_turns=10)
    history = store.get_history(user, "chat", max_turns=10)
    assert len(history) == 20
    assert history[0] == {"role": "user", "content": "q5"}
    assert history[-1] == {"role": "assistant", "content": "a14"}
    (rows,) = store._conn.execute("SELECT COUNT(*) FROM messages").fetchone()
    assert rows == 20  # trimmed in the database, not only on read


def test_raw_user_ids_are_never_stored(tmp_path):
    db = tmp_path / "privacy.db"
    store = Store(str(db))
    raw = "wa:8801712345678"
    user = hash_user_id(raw)
    store.add_turn(user, "chat", "hi", "hello", 10)
    store.set_mode(user, "about")
    store.allow_request(user, 10, 60)
    store.close()
    dump = "\n".join(sqlite3.connect(db).iterdump())
    assert "8801712345678" not in dump
    assert user in dump


def test_clear_history_only_affects_one_user(tmp_path):
    store = Store(str(tmp_path / "c.db"))
    a, b = hash_user_id("a"), hash_user_id("b")
    store.add_turn(a, "chat", "q", "r", 10)
    store.add_turn(b, "chat", "q", "r", 10)
    store.clear_history(a)
    assert store.get_history(a, "chat", 10) == []
    assert len(store.get_history(b, "chat", 10)) == 2


def test_mode_defaults_to_chat_and_persists(tmp_path):
    path = str(tmp_path / "mode.db")
    store = Store(path)
    user = hash_user_id("u")
    assert store.get_mode(user) == "chat"
    store.set_mode(user, "about")
    store.close()
    assert Store(path).get_mode(user) == "about"


def test_mark_processed_dedupes(tmp_path):
    store = Store(str(tmp_path / "d.db"))
    assert store.mark_processed("wamid.1") is True
    assert store.mark_processed("wamid.1") is False
    assert store.mark_processed("wamid.2") is True


def test_rate_limiter_sliding_window(tmp_path):
    clock = FakeClock()
    store = Store(str(tmp_path / "r.db"), clock=clock)
    user = hash_user_id("u")
    assert all(store.allow_request(user, 2, 60) for _ in range(2))
    assert store.allow_request(user, 2, 60) is False
    clock.t += 61
    assert store.allow_request(user, 2, 60) is True


def test_in_memory_database_works():
    store = Store(":memory:")
    store.add_turn("u", "chat", "q", "a", 10)
    assert len(store.get_history("u", "chat", 10)) == 2
