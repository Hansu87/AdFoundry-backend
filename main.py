# main.py
import os, hmac, time, uuid, secrets, hashlib, base64
from typing import Optional, List

from fastapi import FastAPI, Request, Depends, HTTPException, status, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import httpx
import stripe

# ----------------------------- helpers & flags -----------------------------
def _b(v: Optional[str], d: bool = False) -> bool:
    return d if v is None else v.lower() in {"1", "true", "yes", "y", "on"}

WEB_ORIGIN       = os.getenv("WEB_ORIGIN", "http://localhost:3000")
WEB_ORIGIN_REGEX = os.getenv("WEB_ORIGIN_REGEX", "")
AD_API_KEY       = os.getenv("AD_API_KEY", "")
MOCK_MODE        = _b(os.getenv("MOCK_MODE"), True)

# Providers
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY","")
OPENAI_MODEL_TEXT   = os.getenv("OPENAI_MODEL_TEXT","gpt-4o-mini")

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN","")
REPLICATE_IMAGE_MODEL = os.getenv("REPLICATE_IMAGE_MODEL","")  # e.g. black-forest-labs/FLUX.1-schnell:xxx
REPLICATE_VIDEO_MODEL = os.getenv("REPLICATE_VIDEO_MODEL","")
REPLICATE_POLL_SEC  = int(os.getenv("REPLICATE_POLL_SEC","2"))

RESEND_API_KEY  = os.getenv("RESEND_API_KEY","")
RESEND_FROM     = os.getenv("RESEND_FROM","")

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID","")
TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN","")
TWILIO_MESSAGING_SERVICE_SID = os.getenv("TWILIO_MESSAGING_SERVICE_SID","")
TWILIO_FROM       = os.getenv("TWILIO_FROM","")  # E.164, falls kein Messaging Service

# Stripe (Subscriptions + Add-ons)
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY","")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET","")
STRIPE_PRICE_STARTER = os.getenv("STRIPE_PRICE_STARTER","")
STRIPE_PRICE_PRO     = os.getenv("STRIPE_PRICE_PRO","")
STRIPE_PRICE_IMAGE_PACK_100 = os.getenv("STRIPE_PRICE_IMAGE_PACK_100","")
STRIPE_PRICE_VIDEO_PACK_5   = os.getenv("STRIPE_PRICE_VIDEO_PACK_5","")
if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

# Limits & Quotas (in-memory MVP)
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN","10"))
COOLDOWN_SEC       = int(os.getenv("COOLDOWN_SEC","10"))
DAILY_BUDGET_USD   = float(os.getenv("DAILY_BUDGET_USD","10"))

QUOTA_MONTH = {
    "free":    (int(os.getenv("QUOTA_MONTH_FREE","10")),   float(os.getenv("VIDEO_MIN_FREE","0.5"))),
    "starter": (int(os.getenv("QUOTA_MONTH_STARTER","200")), float(os.getenv("VIDEO_MIN_STARTER","1"))),
    "pro":     (int(os.getenv("QUOTA_MONTH_PRO","1000")), float(os.getenv("VIDEO_MIN_PRO","5"))),
}

_last_minute = {}   # {uid:min} -> count
_last_call   = {}   # {uid} -> ts
_spend       = {}   # {YYYYMMDD} -> usd
_usage_img   = {}   # {YYYYMM:uid} -> images used
_usage_video = {}   # {YYYYMM:uid} -> minutes used
PLAN_BY_USER = {}   # {api_key} -> {"plan":"free|starter|pro"}

def _month_key(): return time.strftime("%Y%m")
def _day_key():   return time.strftime("%Y%m%d")

# ----------------------------- FastAPI app & CORS -----------------------------
app = FastAPI(title="AdKiln API", version="1.1.0")

_allowed = [o.strip() for o in WEB_ORIGIN.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed if _allowed else [],
    allow_origin_regex=WEB_ORIGIN_REGEX if WEB_ORIGIN_REGEX else None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------- middleware meta -----------------------------
@app.middleware("http")
async def req_meta(req: Request, call_next):
    rid = req.headers.get("x-request-id") or uuid.uuid4().hex
    t0 = time.time()
    try:
        resp = await call_next(req)
    except Exception as e:
        resp = JSONResponse({"error":"Internal Server Error"}, status_code=500)
    resp.headers["x-request-id"] = rid
    resp.headers["x-response-time"] = f"{int((time.time()-t0)*1000)}ms"
    return resp

# ----------------------------- auth -----------------------------
def get_user(req: Request):
    auth = req.headers.get("authorization") or req.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        return {"sub":"bearer"}
    api_key = req.headers.get("x-api-key")
    if AD_API_KEY and api_key and hmac.compare_digest(api_key, AD_API_KEY):
        return {"sub":"api-key"}
    xut = req.headers.get("x-user-token")
    if xut:
        return {"sub":"x-user-token"}
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Unauthorized")

# ----------------------------- models -----------------------------
class CopyGenReq(BaseModel):
    prompt: str = Field(..., min_length=1)
class CopyGenResp(BaseModel):
    text: str
    id: Optional[str] = None

class PngReq(BaseModel):
    prompt: str = Field(..., min_length=1)
    size: str = "1024x1024"           # kompatibel: Frontend darf es ignorieren
    steps: Optional[int] = None
    guidance: Optional[float] = None
    negative: Optional[str] = None
    seed: Optional[int] = None
class PngResp(BaseModel):
    imageUrl: str
    id: Optional[str] = None

class VideoReq(BaseModel):
    prompt: str = Field(..., min_length=1)
    durationSec: int = 6
    fps: int = 24
    resolution: str = "720p"
class VideoResp(BaseModel):
    videoUrl: str
    id: Optional[str] = None

class EmailReq(BaseModel):
    prompt: str = Field(..., min_length=1)  # für generate/email (Mock/Template)
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

# ----------------------------- mock fns -----------------------------
def _mock_copy(p: str) -> CopyGenResp:
    t = f"Ad Copy: {' '.join(p.split()[:12])} — klare Benefits, Social Proof, 1 CTA."
    return CopyGenResp(text=t, id=secrets.token_hex(6))
def _mock_png(p: str) -> PngResp:
    seed = "-".join(p.split()) or "seed"
    return PngResp(imageUrl=f"https://picsum.photos/seed/{seed}/512/512", id=secrets.token_hex(6))
def _mock_video(p: str) -> VideoResp:
    return VideoResp(videoUrl="https://www.w3schools.com/html/mov_bbb.mp4", id=secrets.token_hex(6))
def _mock_email(p: str) -> EmailResp:
    subj = f"Betreff: {' '.join(p.split()[:5])}" if p else "Beispiel Betreff"
    txt  = f"Dies ist eine Beispiel-E-Mail für: {p}" if p else "Dies ist eine Beispiel-E-Mail."
    return EmailResp(subject=subj, text=txt, id=secrets.token_hex(6))
def _mock_sms(p: str) -> SmsResp:
    msg = f"SMS: {' '.join(p.split()[:8])}…" if p else "Dies ist eine Beispiel-SMS."
    return SmsResp(message=msg, id=secrets.token_hex(6))

# ----------------------------- rate/quota/budget -----------------------------
def _rl_minute(uid: str):
    now_min = int(time.time()//60)
    key = f"{uid}:{now_min}"
    _last_minute[key] = _last_minute.get(key,0) + 1
    if _last_minute[key] > RATE_LIMIT_PER_MIN:
        raise HTTPException(429, "Too many requests")

def _cooldown(uid: str):
    now = int(time.time())
    last = _last_call.get(uid, 0)
    if now - last < COOLDOWN_SEC:
        raise HTTPException(429, f"Wait {COOLDOWN_SEC-(now-last)}s")
    _last_call[uid] = now

def _charge(cost: float):
    day = _day_key()
    _spend[day] = round(_spend.get(day,0.0) + cost, 4)
    if _spend[day] > DAILY_BUDGET_USD:
        raise HTTPException(503, "Daily budget reached")

def _user_id(req: Request, user: dict) -> str:
    return req.headers.get("x-api-key") or user.get("sub") or "anon"
def _plan_for(uid: str) -> str:
    return PLAN_BY_USER.get(uid, {}).get("plan", "free")
def _enforce_image_quota(uid: str):
    plan = _plan_for(uid)
    limit, _ = QUOTA_MONTH[plan]
    key = f"{_month_key()}:{uid}"
    used = _usage_img.get(key, 0)
    if used >= limit:
        raise HTTPException(402, "Image quota exceeded — upgrade plan")
    _usage_img[key] = used + 1
def _enforce_video_quota(uid: str, minutes: float):
    plan = _plan_for(uid)
    _, vlimit = QUOTA_MONTH[plan]
    key = f"{_month_key()}:{uid}"
    used = _usage_video.get(key, 0.0)
    if used + minutes > vlimit + 1e-6:
        raise HTTPException(402, "Video minutes exceeded — upgrade plan")
    _usage_video[key] = round(used + minutes, 3)

# ----------------------------- provider helpers -----------------------------
def _openai_copy(prompt: str) -> str:
    if not OPENAI_API_KEY:
        raise HTTPException(503, "OPENAI_API_KEY missing")
    payload = {
        "model": OPENAI_MODEL_TEXT,
        "messages": [
            {"role":"system","content":"You write concise, persuasive ad copy."},
            {"role":"user","content": f"Write ONE ad copy, max 45 words. Topic: {prompt}"}
        ],
        "temperature": 0.7,
        "max_tokens": 200
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    with httpx.Client(timeout=15) as c:
        r = c.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
    if r.status_code >= 400:
        raise HTTPException(502, f"OpenAI error {r.status_code}: {r.text[:160]}")
    data = r.json()
    return (data.get("choices") or [{}])[0].get("message",{}).get("content","").strip() or "No result."

def _parse_size(s: str):
    try:
        w,h = s.lower().split("x"); return int(w), int(h)
    except: return 1024,1024

def _replicate_start_image(prompt: str, size: str, **kwargs) -> str:
    if not (REPLICATE_API_TOKEN and REPLICATE_IMAGE_MODEL):
        raise HTTPException(503, "Replicate image not configured")
    headers = {"Authorization": f"Token {REPLICATE_API_TOKEN}", "Content-Type":"application/json"}
    w,h = _parse_size(size)
    model_input = {"prompt": prompt, "width": w, "height": h}
    model_input.update({k:v for k,v in kwargs.items() if v is not None})
    body = {"version": REPLICATE_IMAGE_MODEL, "input": model_input}
    with httpx.Client(timeout=15) as c:
        r = c.post("https://api.replicate.com/v1/predictions", json=body, headers=headers)
    if r.status_code >= 400:
        raise HTTPException(502, f"Replicate image error {r.status_code}: {r.text[:180]}")
    return r.json()["id"]

def _replicate_start_video(prompt: str, fps: int, duration: int, resolution: str) -> str:
    if not (REPLICATE_API_TOKEN and REPLICATE_VIDEO_MODEL):
        raise HTTPException(503, "Replicate video not configured")
    headers = {"Authorization": f"Token {REPLICATE_API_TOKEN}", "Content-Type":"application/json"}
    body = {"version": REPLICATE_VIDEO_MODEL, "input": {
        "prompt": prompt, "fps": fps, "duration": duration, "resolution": resolution
    }}
    with httpx.Client(timeout=15) as c:
        r = c.post("https://api.replicate.com/v1/predictions", json=body, headers=headers)
    if r.status_code >= 400:
        raise HTTPException(502, f"Replicate video error {r.status_code}: {r.text[:180]}")
    return r.json()["id"]

def _replicate_poll(pred_id: str) -> dict:
    headers = {"Authorization": f"Token {REPLICATE_API_TOKEN}"}
    with httpx.Client(timeout=10) as c:
        r = c.get(f"https://api.replicate.com/v1/predictions/{pred_id}", headers=headers)
    r.raise_for_status()
    return r.json()

def _watermark_data_url_from_url(png_url: str, label: str = "FREE") -> str:
    # Holt PNG, legt unten eine dunkle Leiste mit Hinweis über das Bild und liefert data:-URL
    try:
        from PIL import Image, ImageDraw
        with httpx.Client(timeout=15) as c:
            raw = c.get(png_url).content
        import io
        im = Image.open(io.BytesIO(raw)).convert("RGBA")
        W,H = im.size
        overlay = Image.new("RGBA", (W,H), (0,0,0,0))
        d = ImageDraw.Draw(overlay)
        bar_h = max(36, H//14)
        d.rectangle([(0, H-bar_h), (W, H)], fill=(0,0,0,120))
        d.text((16, H-bar_h+8), f"{label} – upgrade to remove watermark", fill=(255,255,255,230))
        out = Image.alpha_composite(im, overlay).convert("RGB")
        buf = io.BytesIO(); out.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"
    except Exception:
        return png_url

# ----------------------------- endpoints -----------------------------
@app.get("/api/health")
def health():
    day = _day_key()
    return {"ok": True, "service":"adkiln-api", "version": app.version,
            "mock_mode": MOCK_MODE, "spend_today": _spend.get(day, 0.0)}

# --- Copy ---
@app.post("/api/generate/copy", response_model=CopyGenResp)
def gen_copy(req: CopyGenReq, request: Request, user=Depends(get_user)):
    uid = request.headers.get("x-api-key") or "anon"
    _rl_minute(uid); _cooldown(uid)
    if MOCK_MODE:
        return _mock_copy(req.prompt)
    text = _openai_copy(req.prompt)
    return CopyGenResp(text=text, id=secrets.token_hex(6))

# --- PNG ---
@app.post("/api/generate/png", response_model=PngResp)
def gen_png(req: PngReq, request: Request, user=Depends(get_user)):
    uid = _user_id(request, user)
    _rl_minute(uid); _cooldown(uid); _enforce_image_quota(uid); _charge(0.04)  # grobe Kostenannahme
    if MOCK_MODE:
        return _mock_png(req.prompt)
    # start + poll synchron, damit Contract 200 {imageUrl} bleibt
    pred = _replicate_start_image(req.prompt, req.size, steps=req.steps, guidance=req.guidance,
                                  negative_prompt=req.negative, seed=req.seed)
    t0 = time.time()
    while True:
        time.sleep(REPLICATE_POLL_SEC)
        st = _replicate_poll(pred)
        s = st.get("status")
        if s in ("succeeded","failed","canceled"):
            if s!="succeeded": raise HTTPException(502, f"Replicate status: {s}")
            out = st.get("output")
            url = out if isinstance(out,str) else (out[-1] if isinstance(out,list) else None)
            if not url: raise HTTPException(502, "No image URL from provider")
            if _plan_for(uid) == "free":
                url = _watermark_data_url_from_url(url, "FREE")
            return PngResp(imageUrl=url, id=secrets.token_hex(6))
        if time.time()-t0 > 60:
            raise HTTPException(504, "Image generation timeout")

# --- Video ---
@app.post("/api/generate/video", response_model=VideoResp)
def gen_video(req: VideoReq, request: Request, user=Depends(get_user)):
    uid = _user_id(request, user)
    _rl_minute(uid); _cooldown(uid); _enforce_video_quota(uid, minutes=max(0.1, req.durationSec/60)); _charge(0.12)
    if MOCK_MODE:
        return _mock_video(req.prompt)
    pred = _replicate_start_video(req.prompt, req.fps, req.durationSec, req.resolution)
    t0 = time.time()
    while True:
        time.sleep(REPLICATE_POLL_SEC)
        st = _replicate_poll(pred)
        s = st.get("status")
        if s in ("succeeded","failed","canceled"):
            if s!="succeeded": raise HTTPException(502, f"Replicate status: {s}")
            out = st.get("output")
            url = out if isinstance(out,str) else (out[-1] if isinstance(out,list) else None)
            if not url: raise HTTPException(502, "No video URL from provider")
            return VideoResp(videoUrl=url, id=secrets.token_hex(6))
        if time.time()-t0 > 180:
            raise HTTPException(504, "Video generation timeout")

# --- Generate Email/SMS (mock content) ---
@app.post("/api/generate/email", response_model=EmailResp)
def gen_email(req: EmailReq, request: Request, user=Depends(get_user)):
    uid = _user_id(request, user); _rl_minute(uid); _cooldown(uid)
    return _mock_email(req.prompt) if MOCK_MODE else _mock_email(req.prompt)

@app.post("/api/generate/sms", response_model=SmsResp)
def gen_sms(req: SmsReq, request: Request, user=Depends(get_user)):
    uid = _user_id(request, user); _rl_minute(uid); _cooldown(uid)
    return _mock_sms(req.prompt) if MOCK_MODE else _mock_sms(req.prompt)

# --- Send Email (Resend) ---
@app.post("/api/send/email")
def send_email(body: dict, request: Request, user=Depends(get_user)):
    uid = _user_id(request, user); _rl_minute(uid); _cooldown(uid)
    if MOCK_MODE:
        return {"status":"sent","to": body.get("to")}
    if not (RESEND_API_KEY and RESEND_FROM):
        raise HTTPException(503, "Email provider not configured")
    headers = {"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type":"application/json"}
    payload = {"from": RESEND_FROM, "to": [body.get("to")], "subject": body.get("subject","(No subject)"), "text": body.get("text","")}
    with httpx.Client(timeout=15) as c:
        r = c.post("https://api.resend.com/emails", json=payload, headers=headers)
    if r.status_code >= 400:
        raise HTTPException(502, f"Resend error {r.status_code}: {r.text[:160]}")
    return {"status": "sent", "to": body.get("to")}

# --- Send SMS (Twilio) ---
@app.post("/api/send/sms")
def send_sms(body: dict, request: Request, user=Depends(get_user)):
    uid = _user_id(request, user); _rl_minute(uid); _cooldown(uid)
    if MOCK_MODE:
        return {"status":"sent","to": body.get("to")}
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and (TWILIO_MESSAGING_SERVICE_SID or TWILIO_FROM)):
        raise HTTPException(503, "SMS provider not configured")
    data = {"To": body.get("to"), "Body": body.get("message","")}
    if TWILIO_MESSAGING_SERVICE_SID: data["MessagingServiceSid"] = TWILIO_MESSAGING_SERVICE_SID
    else: data["From"] = TWILIO_FROM
    auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    with httpx.Client(timeout=15, auth=auth) as c:
        r = c.post(f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json", data=data)
    if r.status_code >= 400:
        raise HTTPException(502, f"Twilio error {r.status_code}: {r.text[:160]}")
    return {"status":"sent","to": body.get("to")}

# --- Lists (unchanged contracts) ---
@app.get("/api/creatives", response_model=List[Creative])
def creatives(user=Depends(get_user)):
    return [Creative(id=secrets.token_hex(8), name="Summer Promo Set"),
            Creative(id=secrets.token_hex(8), name="Landing Headlines")]

@app.get("/api/copy-logs", response_model=List[CopyLog])
def copy_logs(user=Depends(get_user)):
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return [CopyLog(id=secrets.token_hex(6), text="Beispiel Werbetext für Black Friday …", createdAt=now)]

# --- Billing (Stripe) ---
@app.post("/api/billing/checkout/{plan}")
def billing_checkout(plan: str, request: Request, user=Depends(get_user)):
    if plan not in ("starter","pro"): raise HTTPException(400, "unknown plan")
    price = STRIPE_PRICE_STARTER if plan == "starter" else STRIPE_PRICE_PRO
    if not (STRIPE_SECRET_KEY and price): raise HTTPException(503, "Stripe not configured")
    api_key = request.headers.get("x-api-key","")
    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": price, "quantity": 1}],
        success_url=str(request.base_url) + "?success=1",
        cancel_url=str(request.base_url) + "?canceled=1",
        metadata={"api_key": api_key, "plan": plan}
    )
    return {"url": session.url}

@app.post("/api/billing/checkout/addon")
def billing_addon(type: str, request: Request, user=Depends(get_user)):
    pid = STRIPE_PRICE_IMAGE_PACK_100 if type=="image100" else (STRIPE_PRICE_VIDEO_PACK_5 if type=="video5" else None)
    if not pid: raise HTTPException(400, "unknown addon")
    if not STRIPE_SECRET_KEY: raise HTTPException(503, "Stripe not configured")
    api_key = request.headers.get("x-api-key","")
    session = stripe.checkout.Session.create(
        mode="payment", line_items=[{"price": pid, "quantity": 1}],
        success_url=str(request.base_url) + "?topup=1",
        cancel_url=str(request.base_url) + "?cancel=1",
        metadata={"api_key": api_key, "addon": type}
    )
    return {"url": session.url}

@app.post("/api/billing/webhook")
async def billing_webhook(request: Request, stripe_signature: str = Header(None)):
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(503, "Webhook secret missing")
    payload = await request.body()
    try:
        event = stripe.Webhook.construct_event(payload, stripe_signature, STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        raise HTTPException(400, f"Webhook error: {e}")
    t = event["type"]; data = event["data"]["object"]; meta = data.get("metadata") or {}
    api_key = meta.get("api_key")
    if not api_key: return {"ok": True}

    if t == "checkout.session.completed":
        plan = meta.get("plan")
        if plan in ("starter","pro"):
            PLAN_BY_USER[api_key] = {"plan": plan}
        addon = meta.get("addon")
        mk = _month_key()
        if addon == "image100":
            # add 100 image credits (wir führen ein separates „Extra“-Feld nicht – MVP)
            key = f"{mk}:{api_key}"
            _usage_img[key] = max(0, _usage_img.get(key, 0) - 100)  # negative Nutzung = Gutschrift
        if addon == "video5":
            key = f"{mk}:{api_key}"
            _usage_video[key] = max(0.0, _usage_video.get(key, 0.0) - 5.0)
    return {"ok": True}

# --- tiny helper for UI ---
@app.get("/api/me/plan")
def me_plan(request: Request, user=Depends(get_user)):
    uid = _user_id(request, user)
    plan = _plan_for(uid)
    mk = _month_key()
    used_i = _usage_img.get(f"{mk}:{uid}", 0)
    used_v = _usage_video.get(f"{mk}:{uid}", 0.0)
    lim_i, lim_v = QUOTA_MONTH[plan]
    return {"plan": plan, "images_used": used_i, "images_limit": lim_i,
            "video_min_used": used_v, "video_min_limit": lim_v}

@app.get("/")
def root():
    return {"message":"AdKiln API up","docs":"/docs"}
