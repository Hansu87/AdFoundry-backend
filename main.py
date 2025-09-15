import os
import hmac
import time
import uuid
import secrets
from typing import Optional, List

from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


def _b(v: Optional[str], d: bool = False) -> bool:
    """Helper to coerce environment variables into booleans.

    Accepts truthy strings like "1", "true", "yes" and returns a boolean.
    If `v` is None, returns the default provided via `d`.
    """
    if v is None:
        return d
    return v.lower() in {"1", "true", "yes", "y", "on"}


# -----------------------------------------------------------------------------
# Configuration via environment variables. Sensible defaults are provided so
# that the API never crashes due to missing configuration.

# Comma‑separated list of allowed origins for CORS. A trailing slash is
# automatically stripped by the proxy in Next.js, so we do the same here.
WEB_ORIGIN = os.getenv("WEB_ORIGIN", "http://localhost:3000")
# Regular expression for matching dynamic origins (e.g. Vercel preview URLs).
WEB_ORIGIN_REGEX = os.getenv("WEB_ORIGIN_REGEX", "")
# API key used for simple authentication. When unset, only Bearer tokens are
# accepted. See get_user() for details.
AD_API_KEY = os.getenv("AD_API_KEY", "")
# Enable mock mode by default. When true, all generation endpoints return
# deterministic sample data instead of calling external providers. This makes
# development and testing reliable and avoids errors when no provider is
# configured.
MOCK_MODE = _b(os.getenv("MOCK_MODE"), True)


# -----------------------------------------------------------------------------
# Application setup

app = FastAPI(title="AdFoundry API", version="1.0.0")

# Configure CORS based on WEB_ORIGIN and WEB_ORIGIN_REGEX. When no origins
# are configured, the service will disallow all cross‑origin requests except
# those matched by the regex. The `allow_headers` contains "*" so that
# Authorization and x-api-key headers are allowed.
_allowed_origins = [o.strip() for o in WEB_ORIGIN.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins if _allowed_origins else [],
    allow_origin_regex=WEB_ORIGIN_REGEX if WEB_ORIGIN_REGEX else None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Middleware for request metadata

@app.middleware("http")
async def req_meta(req: Request, call_next):
    """Attach a request ID and response time to every response.

    A UUID is generated when a request does not provide an "x-request-id"
    header. Additionally, execution time is measured and exposed via
    "x-response-time". Any unhandled exceptions result in a JSON 500 error.
    """
    rid = req.headers.get("x-request-id") or uuid.uuid4().hex
    t0 = time.time()
    try:
        resp = await call_next(req)
    except Exception:
        resp = JSONResponse({"error": "Internal Server Error"}, status_code=500)
    resp.headers["x-request-id"] = rid
    resp.headers["x-response-time"] = f"{int((time.time() - t0) * 1000)}ms"
    return resp


# -----------------------------------------------------------------------------
# Authentication dependency

def get_user(req: Request):
    """Authenticate incoming requests.

    Supports two authentication schemes: Bearer tokens and static API keys. A
    valid Bearer token is identified by the "Authorization" header beginning
    with "Bearer ". For API key authentication the client must send an
    "x-api-key" header equal to AD_API_KEY. Legacy compatibility with a
    "x-user-token" header is maintained. When no valid credentials are
    provided, a 401 Unauthorized error is raised.
    """
    # Bearer authentication
    auth = req.headers.get("authorization") or req.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        # In a real implementation, validate token here and return user info.
        return {"sub": "bearer"}
    # API key authentication via x-api-key header
    api_key = req.headers.get("x-api-key")
    if AD_API_KEY and api_key and hmac.compare_digest(api_key, AD_API_KEY):
        return {"sub": "api-key"}
    # Legacy x-user-token fallback
    xut = req.headers.get("x-user-token")
    if xut:
        return {"sub": "x-user-token"}
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Unauthorized")


# -----------------------------------------------------------------------------
# Pydantic models representing the requests and responses for the API. These
# definitions ensure type safety and consistent JSON shapes.

class CopyGenReq(BaseModel):
    prompt: str = Field(..., min_length=1)


class CopyGenResp(BaseModel):
    text: str
    id: Optional[str] = None


class PngReq(BaseModel):
    prompt: str = Field(..., min_length=1)


class PngResp(BaseModel):
    imageUrl: str
    id: Optional[str] = None


class VideoReq(BaseModel):
    prompt: str = Field(..., min_length=1)


class VideoResp(BaseModel):
    videoUrl: str
    id: Optional[str] = None


class EmailReq(BaseModel):
    prompt: str = Field(..., min_length=1)


class EmailResp(BaseModel):
    subject: str
    text: str
    id: Optional[str] = None


class SmsReq(BaseModel):
    prompt: str = Field(..., min_length=1)


class SmsResp(BaseModel):
    message: str
    id: Optional[str] = None


class Creative(BaseModel):
    id: str
    name: str


class CopyLog(BaseModel):
    id: str
    text: str
    createdAt: str


# -----------------------------------------------------------------------------
# Mock generators

def _mock_copy(prompt: str) -> CopyGenResp:
    """Generate deterministic ad copy for a given prompt.

    Creates a simple German advertisement text by taking the first twelve
    words of the prompt and appending a standard marketing formula. The id
    field is a random hex string for client correlation.
    """
    preview = " ".join(prompt.split()[:12])
    text = f"Ad Copy: {preview} — klare Benefits, Social Proof, 1 CTA."
    return CopyGenResp(text=text, id=secrets.token_hex(6))


def _mock_png(prompt: str) -> PngResp:
    """Generate a placeholder PNG URL for a given prompt.

    Encodes the prompt into a seed used by the picsum.photos service. This
    yields a deterministic image for a given prompt.
    """
    seed = "-".join(prompt.split()) or "seed"
    url = f"https://picsum.photos/seed/{seed}/512/512"
    return PngResp(imageUrl=url, id=secrets.token_hex(6))


def _mock_video(prompt: str) -> VideoResp:
    """Generate a placeholder video URL for a given prompt.

    Uses a fixed MP4 sample hosted by w3schools. In a real implementation
    this would call a video generation provider.
    """
    return VideoResp(videoUrl="https://www.w3schools.com/html/mov_bbb.mp4", id=secrets.token_hex(6))


def _mock_email(prompt: str) -> EmailResp:
    """Generate a placeholder email subject and body.

    The subject is based on the first five words of the prompt. The body
    simply describes that this is a sample email for the given prompt.
    """
    if prompt:
        subject = f"Betreff: {' '.join(prompt.split()[:5])}"
        text = f"Dies ist eine Beispiel‑E‑Mail für: {prompt}"
    else:
        subject = "Beispiel Betreff"
        text = "Dies ist eine Beispiel‑E‑Mail."
    return EmailResp(subject=subject, text=text, id=secrets.token_hex(6))


def _mock_sms(prompt: str) -> SmsResp:
    """Generate a placeholder SMS message.

    Produces a German SMS by truncating the prompt to eight words and
    appending an ellipsis. If no prompt is provided a generic message is
    returned.
    """
    if prompt:
        message = f"SMS: {' '.join(prompt.split()[:8])}…"
    else:
        message = "Dies ist eine Beispiel‑SMS."
    return SmsResp(message=message, id=secrets.token_hex(6))


# -----------------------------------------------------------------------------
# Health check endpoint

@app.get("/api/health")
def health():
    """Return basic health information about the service."""
    return {
        "ok": True,
        "service": "adforge-api",
        "version": app.version,
        "mock_mode": MOCK_MODE,
    }


# -----------------------------------------------------------------------------
# API endpoints

@app.post("/api/generate/copy", response_model=CopyGenResp)
def gen_copy(req: CopyGenReq, user: dict = Depends(get_user)):
    """Generate ad copy via mock or real provider."""
    # Real implementation would call out to an external service here.
    return _mock_copy(req.prompt) if MOCK_MODE else _mock_copy(req.prompt)


@app.post("/api/generate/png", response_model=PngResp)
def gen_png(req: PngReq, user: dict = Depends(get_user)):
    """Generate a PNG via mock or real provider."""
    return _mock_png(req.prompt) if MOCK_MODE else _mock_png(req.prompt)


@app.post("/api/generate/video", response_model=VideoResp)
def gen_video(req: VideoReq, user: dict = Depends(get_user)):
    """Generate a video via mock or real provider."""
    return _mock_video(req.prompt) if MOCK_MODE else _mock_video(req.prompt)


@app.post("/api/generate/email", response_model=EmailResp)
def gen_email(req: EmailReq, user: dict = Depends(get_user)):
    """Generate an email via mock or real provider."""
    return _mock_email(req.prompt) if MOCK_MODE else _mock_email(req.prompt)


@app.post("/api/generate/sms", response_model=SmsResp)
def gen_sms(req: SmsReq, user: dict = Depends(get_user)):
    """Generate an SMS via mock or real provider."""
    return _mock_sms(req.prompt) if MOCK_MODE else _mock_sms(req.prompt)


@app.post("/api/send/email")
def send_email(body: dict, user: dict = Depends(get_user)):
    """Pretend to send an email.

    Returns a success object containing the recipient address. A real
    implementation would queue the email or call an external service.
    """
    return {"status": "sent", "to": body.get("to")}


@app.post("/api/send/sms")
def send_sms(body: dict, user: dict = Depends(get_user)):
    """Pretend to send an SMS.

    Returns a success object containing the recipient number.
    """
    return {"status": "sent", "to": body.get("to")}


@app.get("/api/creatives", response_model=List[Creative])
def creatives(user: dict = Depends(get_user)):
    """List sample creatives.

    Returns a fixed set of creative objects. In a real implementation this
    would query a database or external API.
    """
    return [
        Creative(id=secrets.token_hex(8), name="Summer Promo Set"),
        Creative(id=secrets.token_hex(8), name="Landing Headlines"),
    ]


@app.get("/api/copy-logs", response_model=List[CopyLog])
def copy_logs(user: dict = Depends(get_user)):
    """Return a mock history of generated copy.

    The timestamp is provided in ISO format for client display.
    """
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return [
        CopyLog(id=secrets.token_hex(6), text="Beispiel Werbetext für Black Friday …", createdAt=now_iso)
    ]


@app.get("/")
def root():
    """Root endpoint returns a simple message and link to docs."""
    return {"message": "AdFoundry API up", "docs": "/docs"}