import os, json, pathlib, datetime
from typing import Optional, List
from sqlmodel import SQLModel, Field, Session, create_engine, select




# --- optional Fernet
FERNET_KEY = os.getenv("FERNET_KEY")

try:
    # benennen, damit der Name klar vom Typ getrennt ist
    from cryptography.fernet import Fernet as CFernet  # type: ignore
except Exception:
    CFernet = None  # type: ignore

def _fernet() -> Optional[object]:
    """Gibt eine Fernet-Instanz zurück, falls KEY+Lib vorhanden sind; sonst None."""
    if not FERNET_KEY or CFernet is None:
        return None
    try:
        return CFernet(FERNET_KEY)  # type: ignore
    except Exception:
        return None

# optional Fernet
# Env-Key (einmal oben im File definieren)
# --- Optional Fernet (encryption)
import os
from typing import Optional, Any



def encrypt(plain: str) -> str:
    f = _fernet()
    return plain if not f else f.encrypt(plain.encode("utf-8")).decode("utf-8")  # type: ignore

def decrypt(cipher: str) -> str:
    f = _fernet()
    if not f:
        return cipher
    try:
        return f.decrypt(cipher.encode("utf-8")).decode("utf-8")  # type: ignore
    except Exception:
        return cipher


# Gebe absichtlich Optional[object] zurück, damit Pylance nicht meckert


DATA_DIR = pathlib.Path("adsuite/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_URL = os.getenv("DB_URL", f"sqlite:///{(DATA_DIR / 'adforge.db').as_posix()}")

engine = create_engine(DB_URL, echo=False, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {})

# ---- Models
class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True)
    created_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.utcnow())

class Shop(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True)
    domain: str = Field(index=True)
    api_version: Optional[str] = None
    token_encrypted: Optional[str] = None
    created_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.utcnow())

class CopyLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True)
    title: str
    price: str
    desc: Optional[str] = ""
    brand: Optional[str] = ""
    variants_json: str  # JSON
    created_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.utcnow())

class Creative(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True)
    title: str
    price: str
    brand: Optional[str] = ""
    cta: Optional[str] = "Shop Now"
    file_path: str
    created_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.utcnow())

def get_session():
    with Session(engine) as session:
        yield session

def init_db():
    SQLModel.metadata.create_all(engine)

# ---- helpers


def encrypt(plain: str) -> str:
    f = _fernet()
    if not f:
        return plain
    return f.encrypt(plain.encode("utf-8")).decode("utf-8")  # type: ignore

def decrypt(cipher: str) -> str:
    f = _fernet()
    if not f:
        return cipher
    try:
        return f.decrypt(cipher.encode("utf-8")).decode("utf-8")  # type: ignore
    except Exception:
        return cipher



# ---- domain ops
def upsert_user_by_email(session: Session, email: str) -> User:
    email = email.lower().strip()
    u = session.exec(select(User).where(User.email == email)).first()
    if u:
        return u
    u = User(email=email)
    session.add(u)
    session.commit()
    session.refresh(u)
    return u

def save_shop(session: Session, user_id: int, domain: str, token_plain: str, api_version: Optional[str]):
    enc = encrypt(token_plain) if token_plain else None
    s = Shop(user_id=user_id, domain=domain, token_encrypted=enc, api_version=api_version)
    session.add(s)
    session.commit()
    session.refresh(s)
    return s

def log_copy(session: Session, user_id: int, title: str, price: str, desc: str, brand: str, variants: List[str]):
    cl = CopyLog(user_id=user_id, title=title, price=price, desc=desc, brand=brand, variants_json=json.dumps(variants, ensure_ascii=False))
    session.add(cl)
    session.commit()
    session.refresh(cl)
    return cl

def save_creative(session: Session, user_id: int, title: str, price: str, brand: str, cta: str, file_path: str):
    cr = Creative(user_id=user_id, title=title, price=price, brand=brand, cta=cta, file_path=file_path)
    session.add(cr)
    session.commit()
    session.refresh(cr)
    return cr
