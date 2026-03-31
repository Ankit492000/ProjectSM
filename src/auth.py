"""Upstox OAuth2 authentication flow — token acquisition and caching."""

import json
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
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

    # Upstox tokens expire at 3:30 AM IST next day.
    # Simple heuristic: if cached more than 20 hours ago, consider expired.
    cached_at = data.get("_cached_at", 0)
    if time.time() - cached_at > 20 * 3600:
        return None
    return data


def get_access_token() -> str:
    """Get a valid access token — from cache or by running the auth flow."""
    cached = _load_cached_token()
    if cached and cached.get("access_token"):
        return cached["access_token"]
    return run_auth_flow()


def run_auth_flow() -> str:
    """Run the full OAuth2 flow: open browser, capture callback, exchange code."""
    auth_code_holder = {}

    parsed = urlparse(UPSTOX_REDIRECT_URI)
    port = parsed.port or 5000
    callback_path = parsed.path or "/callback"

    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            qs = parse_qs(urlparse(self.path).query)
            code = qs.get("code", [None])[0]
            if code and self.path.startswith(callback_path):
                auth_code_holder["code"] = code
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"Authorization successful! You can close this tab.")
            else:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Authorization failed.")

        def log_message(self, format, *args):
            pass  # suppress logs

    url = get_auth_url()
    print(f"Opening browser for Upstox login...\n{url}")
    webbrowser.open(url)

    server = HTTPServer(("127.0.0.1", port), CallbackHandler)
    print(f"Waiting for callback on port {port}...")

    while "code" not in auth_code_holder:
        server.handle_request()

    server.server_close()
    code = auth_code_holder["code"]
    print("Authorization code received. Exchanging for token...")

    token_data = exchange_code_for_token(code)
    _save_token(token_data)
    print("Token cached successfully.")
    return token_data["access_token"]


if __name__ == "__main__":
    token = run_auth_flow()
    print(f"Access token: {token[:20]}...")
