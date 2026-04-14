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

from config import CLERK_JWKS_URL, CLERK_PUBLISHABLE_KEY

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

def _expected_issuer() -> str | None:
    """Derive the expected JWT issuer from CLERK_JWKS_URL (strip /.well-known/jwks.json)."""
    if not CLERK_JWKS_URL:
        return None
    from urllib.parse import urlparse
    parsed = urlparse(CLERK_JWKS_URL)
    return f"{parsed.scheme}://{parsed.netloc}"


def _verify_token(token: str) -> dict:
    if not _HAS_JWT:
        raise RuntimeError("PyJWT[crypto] is not installed")

    header = _pyjwt.get_unverified_header(token)
    kid    = header.get("kid")
    if not kid:
        raise ValueError("JWT header missing 'kid'")

    jwk        = _get_jwk_for_kid(kid)
    public_key = _pyjwt.algorithms.RSAAlgorithm.from_jwk(jwk)

    issuer = _expected_issuer()
    decode_options = {"verify_aud": False}
    decode_kwargs = dict(
        algorithms=["RS256"],
        options=decode_options,
    )
    if issuer:
        decode_kwargs["issuer"] = issuer

    return _pyjwt.decode(token, public_key, **decode_kwargs)


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
        log.warning("Token verification failed: %s", exc)
        raise HTTPException(status_code=401, detail="Invalid or expired token") from exc

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Token has no subject claim")

    return user_id
