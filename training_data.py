# Real Python training corpus
# 30 snippets covering production patterns (FastAPI, Pydantic, standard library)

RAW_PY_CORPUS = []

# ── Dataclasses & Pydantic ─────────────────────────────────────────────────
RAW_PY_CORPUS.append("""from pydantic import BaseModel, Field, EmailStr
from datetime import datetime
from typing import Optional, List

class User(BaseModel):
    id: str = Field(default_factory=lambda: "uuid-placeholder")
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    age: Optional[int] = Field(None, ge=0, le=150)
    role: str = "user"
    tags: List[str] = Field(default_factory=list, max_items=10)
    created_at: datetime = Field(default_factory=datetime.utcnow)""")

RAW_PY_CORPUS.append("""from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class PaginationMeta:
    page: int
    limit: int
    total: int
    
@dataclass
class ApiResponse:
    data: Any
    status: int = 200
    message: str = "OK"
    meta: Optional[PaginationMeta] = None

def create_response(data: Any, status: int = 200) -> ApiResponse:
    return ApiResponse(data=data, status=status)""")

# ── FastAPI / HTTP ────────────────────────────────────────────────────────
RAW_PY_CORPUS.append("""from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel

router = APIRouter(prefix="/users", tags=["users"])

class UserCreate(BaseModel):
    name: str
    email: str
    role: str = "user"

@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate, db_session=Depends(get_db)):
    if not user.name or not user.email:
        raise HTTPException(status_code=400, detail="name and email required")
    
    try:
        new_user = await UserService.create(db_session, user.model_dump())
        return {"data": new_user, "status": 201, "message": "OK"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))""")

RAW_PY_CORPUS.append("""from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.route("/api/data")
@limiter.limit("5/minute")
async def get_data(request: Request):
    return {"data": "You have access"}
    
class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Custom rate logic here
        response = await call_next(request)
        return response""")

# ── Async / await patterns ────────────────────────────────────────────────
RAW_PY_CORPUS.append("""import asyncio
import logging

logger = logging.getLogger(__name__)

async def with_retry(coro_func, *args, retries=3, delay=1.0, backoff=2.0, **kwargs):
    last_error = None
    for attempt in range(retries):
        try:
            return await coro_func(*args, **kwargs)
        except Exception as err:
            last_error = err
            if attempt < retries - 1:
                sleep_time = delay * (backoff ** attempt)
                logger.warning(f"Attempt {attempt+1} failed. Retrying in {sleep_time}s")
                await asyncio.sleep(sleep_time)
                
    logger.error("All retries exhausted")
    raise last_error""")

RAW_PY_CORPUS.append("""import asyncio
from typing import List, Callable, Awaitable, Any

async def run_parallel(tasks: List[Callable[[], Awaitable[Any]]], concurrency: int = 5) -> List[Any]:
    semaphore = asyncio.Semaphore(concurrency)
    
    async def _worker(task):
        async with semaphore:
            return await task()
            
    coroutines = [_worker(t) for t in tasks]
    return await asyncio.gather(*coroutines)""")

# ── Error handling ──────────────────────────────────────────────────────────
RAW_PY_CORPUS.append("""class AppError(Exception):
    def __init__(self, message: str, status_code: int = 500, code: str = "INTERNAL_ERROR", details=None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.details = details or {}

class NotFoundError(AppError):
    def __init__(self, message: str):
        super().__init__(message, status_code=404, code="NOT_FOUND")

class ValidationError(AppError):
    def __init__(self, message: str, details=None):
        super().__init__(message, status_code=400, code="VALIDATION_ERROR", details=details)""")

# ── Classes & OOP ─────────────────────────────────────────────────────────
RAW_PY_CORPUS.append("""from typing import TypeVar, Generic, Optional, List
from abc import ABC, abstractmethod

T = TypeVar('T')

class Repository(Generic[T], ABC):
    @abstractmethod
    async def find_by_id(self, id: str) -> Optional[T]:
        pass
        
    @abstractmethod
    async def find_all(self, **filters) -> List[T]:
        pass
        
    @abstractmethod
    async def save(self, entity: T) -> T:
        pass
        
    @abstractmethod
    async def delete(self, id: str) -> None:
        pass""")

RAW_PY_CORPUS.append("""from collections import OrderedDict
from typing import Generic, TypeVar, Optional

K = TypeVar('K')
V = TypeVar('V')

class LRUCache(Generic[K, V]):
    def __init__(self, capacity: int):
        self.cache: OrderedDict[K, V] = OrderedDict()
        self.capacity = capacity

    def get(self, key: K) -> Optional[V]:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: K, value: V) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)""")

# ── Decorators ────────────────────────────────────────────────────────────
RAW_PY_CORPUS.append("""import functools
import time

def timer_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__} in {run_time:.4f} secs")
        return result
    return wrapper

def async_timer_decorator(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Finished {func.__name__} in {end_time - start_time:.4f} secs")
        return result
    return wrapper""")

# ── Itertools / Utility ──────────────────────────────────────────────────
RAW_PY_CORPUS.append("""import itertools
from typing import Iterable, TypeVar, Dict, List, Callable

T = TypeVar('T')
K = TypeVar('K')

def group_by(items: Iterable[T], key_func: Callable[[T], K]) -> Dict[K, List[T]]:
    # items must be sorted by key_func for itertools.groupby to work properly
    sorted_items = sorted(items, key=key_func)
    grouped = itertools.groupby(sorted_items, key=key_func)
    return {k: list(v) for k, v in grouped}

def chunk_list(lst: List[T], size: int) -> List[List[T]]:
    return [lst[i:i + size] for i in range(0, len(lst), size)]""")

RAW_PY_CORPUS.append("""def deep_merge(target: dict, source: dict) -> dict:
    result = dict(target)
    for key, value in source.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)""")

# ── SQLAlchemy / DB ─────────────────────────────────────────────────────
RAW_PY_CORPUS.append("""from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Tuple, List, Type, Any

async def paginate(
    session: AsyncSession, 
    model: Type[Any], 
    page: int = 1, 
    limit: int = 20, 
    **filters
) -> Tuple[List[Any], int]:
    skip = (page - 1) * limit
    
    query = select(model).filter_by(**filters)
    count_query = select(func.count()).select_from(query.subquery())
    
    paginated_query = query.offset(skip).limit(limit)
    
    total = await session.scalar(count_query)
    result = await session.scalars(paginated_query)
    
    return result.all(), total""")

RAW_PY_CORPUS.append("""from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy import String, Integer, DateTime
from datetime import datetime
import uuid

Base = declarative_base()

class UserModel(Base):
    __tablename__ = "users"
    
    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}""")

# ── Celery / Queue ──────────────────────────────────────────────────────────
RAW_PY_CORPUS.append("""from celery import Celery
import smtplib
from email.message import EmailMessage

app = Celery('email_tasks', broker='redis://localhost:6379/0')

@app.task(bind=True, max_retries=3, default_retry_delay=60)
def send_email_task(self, to_email: str, subject: str, body: str):
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = "noreply@example.com"
        msg['To'] = to_email

        # In a real app, use configured SMTP settings
        print(f"Simulating email send to {to_email}")
        return True
    except Exception as exc:
        # Retry mechanism
        raise self.retry(exc=exc)""")

# ── Context Managers ───────────────────────────────────────────────────────
RAW_PY_CORPUS.append("""from contextlib import contextmanager
import time
import os

@contextmanager
def temporary_env(env_vars):
    # Save original values
    original_env = {key: os.environ.get(key) for key in env_vars}
    
    # Set new values
    os.environ.update(env_vars)
    
    try:
        yield
    finally:
        # Restore original values
        for key, val in original_env.items():
            if val is None:
                del os.environ[key]
            else:
                os.environ[key] = val""")

RAW_PY_CORPUS.append("""from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

class DatabaseManager:
    def __init__(self, session_maker: async_sessionmaker[AsyncSession]):
        self.session_maker = session_maker
        
    async def __aenter__(self) -> AsyncSession:
        self.session = self.session_maker()
        return self.session
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            await self.session.rollback()
        else:
            await self.session.commit()
        await self.session.close()""")

# ── Pathlib & File IO ───────────────────────────────────────────────────
RAW_PY_CORPUS.append("""from pathlib import Path
import json
import logging

def process_directory(dir_path: str, extension: str = ".json") -> list:
    path = Path(dir_path)
    if not path.exists() or not path.is_dir():
        logging.error(f"Directory {dir_path} does not exist")
        return []
        
    results = []
    # rglob recursively searches through all subdirectories
    for file_path in path.rglob(f"*{extension}"):
        try:
            with file_path.open('r', encoding='utf-8') as f:
                data = json.load(f)
                results.append({"file": file_path.name, "data": data})
        except Exception as e:
            logging.error(f"Failed to read {file_path}: {e}")
            
    return results""")

RAW_PY_CORPUS.append("""import aiofiles
import asyncio

async def read_large_file(file_path: str, chunk_size: int = 1024 * 1024):
    chunks = []
    try:
        async with aiofiles.open(file_path, mode='rb') as f:
            while True:
                chunk = await f.read(chunk_size)
                if not chunk:
                    break
                chunks.append(chunk)
                
                # Yield control to event loop for very large files
                await asyncio.sleep(0)
    except FileNotFoundError:
        print(f"File {file_path} not found")
        
    return b''.join(chunks)""")

# ── JWT / Auth ────────────────────────────────────────────────────────────
RAW_PY_CORPUS.append("""from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional

SECRET_KEY = "your-super-secret-key-do-not-hardcode"
ALGORITHM = "HS256"

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
        
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt""")

RAW_PY_CORPUS.append("""from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    user = get_user(db, username=username)
    if user is None:
        raise credentials_exception
    return user""")

# ── Pytest / Testing ───────────────────────────────────────────────────────
RAW_PY_CORPUS.append("""import pytest
from httpx import AsyncClient
from fastapi import FastAPI

@pytest.fixture
async def async_client(app: FastAPI):
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.mark.asyncio
async def test_create_user(async_client: AsyncClient, db_session):
    payload = {"name": "Test User", "email": "test@example.com"}
    response = await async_client.post("/users/", json=payload)
    
    assert response.status_code == 201
    data = response.json()
    assert data["data"]["name"] == "Test User"
    assert "id" in data["data"]""")

RAW_PY_CORPUS.append("""from unittest.mock import AsyncMock, MagicMock
import pytest

@pytest.mark.asyncio
async def test_user_service_create():
    # Arrange
    mock_repo = MagicMock()
    mock_repo.save = AsyncMock(return_value={"id": "1", "name": "John"})
    
    service = UserService(mock_repo)
    
    # Act
    result = await service.create({"name": "John", "email": "john@test.com"})
    
    # Assert
    assert result["id"] == "1"
    mock_repo.save.assert_called_once()
    saved_entity = mock_repo.save.call_args[0][0]
    assert saved_entity["name"] == "John" """)

RAW_TS_CORPUS = RAW_PY_CORPUS  # Keep variable name same as script uses it

print(f"Loaded {len(RAW_PY_CORPUS)} real Python training samples")
print(f"Total characters: {sum(len(s) for s in RAW_PY_CORPUS):,}")
print(f"Sample[0] preview:\\n{RAW_PY_CORPUS[0][:150]}")
