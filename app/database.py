
import os
import re
import hmac
import hashlib
import secrets
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")

PBKDF2_ITERATIONS = 310000

_supabase = None


def get_connection():
    """Return the Supabase client."""

    global _supabase

    if _supabase is None:
        if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
            raise RuntimeError(
                "Supabase credentials are missing. "
                "Check SUPABASE_URL and SUPABASE_SECRET_KEY in .env."
            )

        _supabase = create_client(
            SUPABASE_URL,
            SUPABASE_SECRET_KEY
        )

    return _supabase


def initialize_database():
    """Initialize and verify the Supabase connection."""

    client = get_connection()

    # Verify that the users table is accessible.
    client.table("users").select("id").limit(1).execute()


# --------------------------------------------------
# PASSWORD HASHING
# --------------------------------------------------

def _hash_password(password):
    salt = secrets.token_bytes(16)

    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
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

        return hmac.compare_digest(
            digest.hex(),
            digest_hex
        )

    except (ValueError, TypeError, AttributeError):
        return False


# --------------------------------------------------
# PUBLIC USER DATA
# --------------------------------------------------

def _public_user(row):
    if row is None:
        return None

    return {
        "id": row["id"],
        "name": row["name"],
        "email": row["email"],
        "emergency_email": row["emergency_email"],
    }


# --------------------------------------------------
# USER REGISTRATION
# --------------------------------------------------

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
        raise ValueError(
            "Please enter a valid emergency contact email."
        )

    if len(password) < 8:
        raise ValueError(
            "Password must contain at least 8 characters."
        )

    password_hash = _hash_password(password)

    client = get_connection()

    user_data = {
        "name": name,
        "email": email,
        "password_hash": password_hash,
        "emergency_email": emergency_email,
    }

    try:
        response = (
            client.table("users")
            .insert(user_data)
            .execute()
        )

        if not response.data:
            raise RuntimeError(
                "Supabase did not return the newly created user."
            )

        return _public_user(response.data[0])

    except Exception as error:

        error_code = getattr(error, "code", None)
        error_message = str(error).lower()

        if (
            error_code == "23505"
            or "duplicate key" in error_message
            or "unique constraint" in error_message
        ):
            raise ValueError(
                "An account with this email already exists."
            ) from error

        raise RuntimeError(
            f"Supabase registration failed: {error}"
        ) from error


# --------------------------------------------------
# USER LOGIN
# --------------------------------------------------

def authenticate_user(email, password):

    email = email.strip().lower()

    client = get_connection()

    response = (
        client.table("users")
        .select("*")
        .eq("email", email)
        .limit(1)
        .execute()
    )

    rows = response.data

    if not rows:
        return None

    row = rows[0]

    if not _verify_password(
        password,
        row["password_hash"]
    ):
        return None

    return _public_user(row)


# --------------------------------------------------
# GET USER BY ID
# --------------------------------------------------

def get_user_by_id(user_id):

    client = get_connection()

    response = (
        client.table("users")
        .select("*")
        .eq("id", user_id)
        .limit(1)
        .execute()
    )

    rows = response.data

    if not rows:
        return None

    return _public_user(rows[0])
