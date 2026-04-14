"""
Clerk JWT authentication for FastAPI.

If CLERK_JWKS_URL is not set in .env, the app runs in single-user dev mode
and every request is attributed to the synthetic user id 'dev_user'.

To enable multi-user mode:
  1. Create an application at https://clerk.com
  2. Add to backend/.env:
       CLERK_PUBLISHABLE_KEY=pk_live_...
       CLERK_SECRET_KEY=sk_live_...
       CLERK_JWKS_URL=https://<your-instance>.clerk.accounts.dev/.well-known/jwks.json
"""
import logging
from functools import lru_cache

import httpx
from fastapi import Header, HTTPException

from config import CLERK_JWKS_URL

log = logging.getLogger(__name__)

try:
    import jwt as _pyjwt
    _HAS_JWT = True
except ImportError:
    _HAS_JWT = False
    log.warning("PyJWT not installed — install PyJWT[crypto] to enable Clerk auth")


# ── JWKS helpers ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _fetch_jwks() -> dict:
    """Fetch and cache Clerk's public key set."""
    resp = httpx.get(CLERK_JWKS_URL, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _get_jwk_for_kid(kid: str) -> dict:
    """Return the JWK matching the given key id, refreshing cache once on miss."""
    jwks = _fetch_jwks()
    match = next((k for k in jwks.get("keys", []) if k.get("kid") == kid), None)
    if match:
        return match
    # Cache may be stale — clear and retry once
    _fetch_jwks.cache_clear()
    jwks = _fetch_jwks()
    match = next((k for k in jwks.get("keys", []) if k.get("kid") == kid), None)
    if not match:
        raise ValueError(f"No JWKS key found for kid={kid!r}")
    return match


# ── Token verification ────────────────────────────────────────────────────────

def _verify_token(token: str) -> dict:
    if not _HAS_JWT:
        raise RuntimeError("PyJWT[crypto] is not installed")

    header = _pyjwt.get_unverified_header(token)
    kid    = header.get("kid")
    if not kid:
        raise ValueError("JWT header missing 'kid'")

    jwk        = _get_jwk_for_kid(kid)
    public_key = _pyjwt.algorithms.RSAAlgorithm.from_jwk(jwk)

    return _pyjwt.decode(
        token,
        public_key,
        algorithms=["RS256"],
        options={"verify_aud": False},   # Clerk omits aud in session tokens
    )


# ── FastAPI dependency ────────────────────────────────────────────────────────

def get_user_id(authorization: str = Header(default=None)) -> str:
    """
    FastAPI dependency that resolves the Clerk user id from the
    'Authorization: Bearer <session_token>' header.

    Returns 'dev_user' when Clerk is not configured (no CLERK_JWKS_URL),
    allowing the app to work out-of-the-box without an account system.
    """
    # Dev / single-user mode — Clerk not configured
    if not CLERK_JWKS_URL:
        return "dev_user"

    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Empty bearer token")

    try:
        payload = _verify_token(token)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}") from exc

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token has no subject claim")

    return user_id
