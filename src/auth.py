"""Upstox OAuth2 authentication — simple token acquisition and caching."""

import json
import time
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import httpx
from dotenv import load_dotenv
import os

load_dotenv()

TOKEN_CACHE = Path(os.getenv("UPSTOX_TOKEN_CACHE", ".upstox_token.json"))

UPSTOX_API_KEY = os.getenv("UPSTOX_API_KEY", "")
UPSTOX_API_SECRET = os.getenv("UPSTOX_API_SECRET", "")
UPSTOX_REDIRECT_URI = os.getenv("UPSTOX_REDIRECT_URI", "http://localhost:5000/callback")

AUTH_URL = "https://api.upstox.com/v2/login/authorization/dialog"
TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"


def get_auth_url() -> str:
    """Generate the Upstox OAuth2 authorization URL."""
    return (
        f"{AUTH_URL}?response_type=code"
        f"&client_id={UPSTOX_API_KEY}"
        f"&redirect_uri={UPSTOX_REDIRECT_URI}"
    )


def exchange_code_for_token(code: str) -> dict:
    """Exchange authorization code for access token."""
    resp = httpx.post(
        TOKEN_URL,
        data={
            "code": code,
            "client_id": UPSTOX_API_KEY,
            "client_secret": UPSTOX_API_SECRET,
            "redirect_uri": UPSTOX_REDIRECT_URI,
            "grant_type": "authorization_code",
        },
    )
    resp.raise_for_status()
    return resp.json()


def _save_token(token_data: dict) -> None:
    """Cache token to local file."""
    token_data["_cached_at"] = time.time()
    TOKEN_CACHE.write_text(json.dumps(token_data, indent=2))


def _load_cached_token() -> dict | None:
    """Load cached token if it exists and hasn't expired (valid until 3:30 AM next day)."""
    if not TOKEN_CACHE.exists():
        return None
    try:
        data = json.loads(TOKEN_CACHE.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    cached_at = data.get("_cached_at", 0)
    if time.time() - cached_at > 20 * 3600:
        return None
    return data


def get_access_token() -> str:
    """Get a valid access token -- from cache, .env, or interactive prompt."""
    # 1. Check cache file
    cached = _load_cached_token()
    if cached and cached.get("access_token"):
        return cached["access_token"]

    # 2. Check .env for a directly pasted token
    env_token = os.getenv("UPSTOX_ACCESS_TOKEN", "").strip()
    if env_token:
        _save_token({"access_token": env_token})
        return env_token

    # 3. Interactive prompt
    return run_auth_flow()


def run_auth_flow() -> str:
    """Simple 3-option auth flow. No local server, no blocking."""
    url = get_auth_url()

    print(f"\n{'='*60}")
    print("  UPSTOX LOGIN")
    print(f"{'='*60}")
    print()
    print("  Choose one:")
    print()
    print("  [1] I have an access token (paste it directly)")
    print("      Get it from: Upstox Developer Console > Your App")
    print()
    print("  [2] I have a auth code / redirect URL")
    print(f"      Login URL: {url}")
    print("      After login, Upstox redirects to a URL with ?code=XXXXX")
    print("      Copy that code or the full URL and paste it here")
    print()
    print("  [3] Set token in .env and skip this")
    print("      Add UPSTOX_ACCESS_TOKEN=your_token to .env file")
    print()
    print(f"{'='*60}")

    choice = input("\nChoose [1/2/3]: ").strip()

    if choice == "1":
        token = input("Paste access token: ").strip()
        if not token:
            raise ValueError("No token provided.")
        _save_token({"access_token": token})
        print(f"Token saved to {TOKEN_CACHE}")
        return token

    elif choice == "2":
        user_input = input("Paste redirect URL or auth code: ").strip()

        # Extract code from URL if needed
        if "code=" in user_input:
            qs = parse_qs(urlparse(user_input).query)
            code = qs.get("code", [None])[0]
            if not code:
                raise ValueError("Could not extract 'code' from URL.")
        else:
            code = user_input

        if not code:
            raise ValueError("No auth code provided.")

        print("Exchanging code for token...")
        token_data = exchange_code_for_token(code)
        _save_token(token_data)
        print(f"Token saved to {TOKEN_CACHE}")
        return token_data["access_token"]

    elif choice == "3":
        print(f"\nAdd this line to your .env file:")
        print(f"  UPSTOX_ACCESS_TOKEN=<your_token_here>")
        print(f"\nThen run the command again.")
        raise SystemExit(0)

    else:
        raise ValueError(f"Invalid choice: {choice}")
