import os
import re
import hmac
import hashlib
import secrets
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.getenv(
    "DRIVEGUARD_DB_PATH",
    str(ROOT / "data" / "driveguard_users.db")
))

PBKDF2_ITERATIONS = 310000


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                emergency_email TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()


def _hash_password(password):
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt,
        PBKDF2_ITERATIONS
    )
    return (
        f"pbkdf2_sha256${PBKDF2_ITERATIONS}$"
        f"{salt.hex()}${digest.hex()}"
    )


def _verify_password(password, stored):
    try:
        algorithm, iterations, salt_hex, digest_hex = stored.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(iterations)
        )
        return hmac.compare_digest(digest.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def _public_user(row):
    if row is None:
        return None
    return {
        "id": row["id"],
        "name": row["name"],
        "email": row["email"],
        "emergency_email": row["emergency_email"],
    }


def register_user(name, email, password, emergency_email):
    name = name.strip()
    email = email.strip().lower()
    emergency_email = emergency_email.strip().lower()

    if not name:
        raise ValueError("Please enter your name.")

    email_pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    if not re.match(email_pattern, email):
        raise ValueError("Please enter a valid account email.")

    if not re.match(email_pattern, emergency_email):
        raise ValueError("Please enter a valid emergency contact email.")

    if len(password) < 8:
        raise ValueError("Password must contain at least 8 characters.")

    password_hash = _hash_password(password)

    try:
        with get_connection() as conn:
            conn.execute("""
                INSERT INTO users
                    (name, email, password_hash, emergency_email, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                name,
                email,
                password_hash,
                emergency_email,
                datetime.now(timezone.utc).isoformat()
            ))
            conn.commit()
    except sqlite3.IntegrityError:
        raise ValueError("An account with this email already exists.")

    return authenticate_user(email, password)


def authenticate_user(email, password):
    email = email.strip().lower()

    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?",
            (email,)
        ).fetchone()

    if row is None or not _verify_password(password, row["password_hash"]):
        return None

    return _public_user(row)


def get_user_by_id(user_id):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?",
            (user_id,)
        ).fetchone()
    return _public_user(row)
