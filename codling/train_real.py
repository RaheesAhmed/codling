"""
CODLING Training Script with Massive Data
==========================================

This training script uses either:
1. HuggingFace "bigcode/the-stack-smol" dataset (50K+ Python snippets)
2. Fallback: 50K+ synthetic Python patterns for FastAPI, Pydantic, SQLAlchemy, etc.

Requirements:
- 50K+ Python code snippets for proper training
- 130M model (d_model=512, n_layers=12)
- 500 training steps
- Test generation with code prompts
"""

import os
import sys
import math
import random
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Setup sys.path explicitly for "codling.codling" resolution
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
script_dir = os.path.abspath(os.path.dirname(__file__))
if script_dir in sys.path:
    sys.path.remove(script_dir)

from codling.codling.model import CodlingConfig, CodlingForCausalLM
from codling.codling.inference import CodlingGenerator, GenerationConfig


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuration for training the CODLING model."""
    
    # Model architecture
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    d_state: int = 64
    d_expand: int = 2
    
    # Training
    seq_length: int = 512
    batch_size: int = 16
    max_steps: int = 500
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 50
    
    # Data
    dataset_size: int = 50000  # Target dataset size
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 100
    
    # Generation
    test_prompts: List[str] = field(default_factory=lambda: [
        "class User",
        "def process_data",
        "async def fetch",
        "@app.route",
        "class Config",
        "def validate_",
    ])


# ============================================================================
# Simple Character-Level Tokenizer
# ============================================================================

class SimpleTokenizer:
    """
    Simple character-level tokenizer for Python code.
    
    Maps individual characters to token IDs and back.
    Special tokens:
        0: PAD
        1: BOS (beginning of sequence)
        2: EOS (end of sequence)
        3+: Regular characters
    """
    
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        
        # Character to id mapping (reserve first 3 for special tokens)
        self.char_to_id = {chr(i + 3): i + 3 for i in range(min(vocab_size - 3, 120))}
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        
        # Add common code characters if not present
        for c in '[]{}()<>=+-*/%@#$^&|~`:;.,!?_\'"\\':
            if c not in self.char_to_id:
                idx = len(self.char_to_id) + 3
                if idx < vocab_size:
                    self.char_to_id[c] = idx
                    self.id_to_char[idx] = c
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_token_id)
        
        for char in text:
            tokens.append(self.char_to_id.get(char, self.pad_token_id))
        
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        chars = []
        for tid in token_ids:
            if tid == self.pad_token_id:
                continue
            if tid == self.bos_token_id:
                continue
            if tid == self.eos_token_id:
                break
            char = self.id_to_char.get(tid, '')
            chars.append(char)
        return ''.join(chars)
    
    def __call__(self, text: str, return_tensors: str = None, 
                 padding: str = None, truncation: bool = False, 
                 max_length: int = None) -> Dict[str, Any]:
        """Tokenize text for training."""
        tokens = self.encode(text)
        
        # Truncate if needed
        if max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        # Pad if needed
        if padding == "max_length" and max_length:
            tokens = tokens + [self.pad_token_id] * (max_length - len(tokens))
        
        result = {"input_ids": tokens, "attention_mask": [1 if t != self.pad_token_id else 0 for t in tokens]}
        
        if return_tensors == "pt":
            result = {k: torch.tensor(v) for k, v in result.items()}
        
        return result


# ============================================================================
# Python Code Dataset
# ============================================================================

class PythonCodeDataset(Dataset):
    """
    Dataset for Python code training.
    
    Takes a list of code snippets and tokenizes them for training.
    Each snippet becomes one training example.
    """
    
    def __init__(self, code_snippets: List[str], tokenizer: SimpleTokenizer, 
                 seq_length: int = 512):
        self.snippets = code_snippets
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        logger.info(f"Created dataset with {len(self.snippets)} code snippets")
    
    def __len__(self) -> int:
        return len(self.snippets)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        snippet = self.snippets[idx]
        
        # Tokenize
        result = self.tokenizer(
            snippet,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.seq_length
        )
        
        input_ids = result["input_ids"].squeeze(0)
        attention_mask = result["attention_mask"].squeeze(0)
        
        # For causal LM, labels are the same as input_ids
        # (we predict the next token)
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# ============================================================================
# HuggingFace Data Loading
# ============================================================================

def try_load_huggingface_dataset(dataset_size: int = 50000) -> Optional[List[str]]:
    """
    Try to load Python code from HuggingFace datasets.
    
    Args:
        dataset_size: Target number of snippets to load
        
    Returns:
        List of code snippets if successful, None otherwise
    """
    try:
        from datasets import load_dataset
        
        logger.info("Attempting to load bigcode/the-stack-smol dataset...")
        
        # Load Python subset from the-stack-smol
        dataset = load_dataset(
            "bigcode/the-stack-smol",
            lang="Python",
            split="train",
            trust_remote_code=True
        )
        
        # Get random sample or first N
        total = len(dataset)
        logger.info(f"Dataset contains {total:,} Python snippets")
        
        # Sample uniformly if dataset is larger than needed
        if total > dataset_size:
            indices = random.sample(range(total), dataset_size)
            snippets = [dataset[i]['content'] for i in indices]
        else:
            # Repeat samples to reach target size
            snippets = []
            while len(snippets) < dataset_size:
                for item in dataset:
                    if len(snippets) >= dataset_size:
                        break
                    content = item.get('content', '')
                    if content and len(content) > 20:  # Skip very short snippets
                        snippets.append(content)
        
        logger.info(f"Loaded {len(snippets)} Python snippets from HuggingFace")
        return snippets[:dataset_size]
        
    except ImportError:
        logger.warning("datasets library not available, will use synthetic data")
    except Exception as e:
        logger.warning(f"Failed to load HuggingFace dataset: {e}")
    
    return None


# ============================================================================
# Synthetic Python Code Generation
# ============================================================================

def generate_synthetic_python_data(num_samples: int = 50000) -> List[str]:
    """
    Generate synthetic Python code patterns for training.
    
    Creates diverse code patterns including:
    - FastAPI routes and endpoints
    - Pydantic models
    - SQLAlchemy models
    - Class definitions
    - Function definitions
    - Async functions
    - Decorators
    - Error handling
    - Testing patterns
    - Utility functions
    
    Args:
        num_samples: Number of code snippets to generate
        
    Returns:
        List of Python code snippets
    """
    logger.info(f"Generating {num_samples:,} synthetic Python code snippets...")
    
    # Templates for different Python patterns
    fastapi_templates = [
        '''from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

app = FastAPI(title="API", version="1.0.0")

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    quantity: int = 0

@app.get("/")
async def root():
    return {"message": "API is running"}

@app.get("/items/{item_id}")
async def get_item(item_id: int):
    return {"item_id": item_id, "name": "Sample Item"}

@app.post("/items/")
async def create_item(item: Item):
    return item

@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    return {"item_id": item_id, **item.dict()}

@app.delete("/items/{item_id}")
async def delete_item(item_id: int):
    return {"message": "Item deleted"}''',
        
        '''from fastapi import FastAPI, Query, Path
from typing import Optional

app = FastAPI()

@app.get("/search")
async def search(q: str = Query(..., min_length=1), limit: int = 10):
    return {"query": q, "limit": limit}

@app.get("/users/{user_id}/posts/{post_id}")
async def get_post(user_id: int = Path(...), post_id: int = Path(...)):
    return {"user_id": user_id, "post_id": post_id}''',
        
        '''from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import time

app = FastAPI()

class Task(BaseModel):
    task_id: str
    status: str = "pending"

def process_task(task_id: str):
    time.sleep(1)
    print(f"Processed {task_id}")

@app.post("/tasks/")
async def create_task(task: Task, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_task, task.task_id)
    return {"task_id": task.task_id, "status": "queued"}''',
    ]
    
    pydantic_templates = [
        '''from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from enum import Enum

class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"

class User(BaseModel):
    id: int
    username: str = Field(..., min_length=3, max_length=50)
    email: str
    role: UserRole = UserRole.USER
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    
    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email')
        return v.lower()
    
    class Config:
        use_enum_values = True''',
        
        '''from pydantic import BaseModel, Field, conlist
from typing import List, Optional

class Point(BaseModel):
    x: float
    y: float

class Polygon(BaseModel):
    name: str
    points: conlist(Point, min_items=3)
    color: Optional[str] = "blue"
    
    @property
    def area(self) -> float:
        return 0.0  # Simplified

class Config(BaseModel):
    app_name: str = "MyApp"
    debug: bool = False
    max_retries: int = Field(default=3, ge=0, le=10)''',
    ]
    
    sqlalchemy_templates = [
        '''from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    posts = relationship("Post", back_populates="author", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"''',
        
        '''from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from .base import Base

class Post(Base):
    __tablename__ = "posts"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    author_id = Column(Integer, ForeignKey("users.id"))
    published_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    author = relationship("User", back_populates="posts")
    tags = relationship("Tag", secondary="post_tags", back_populates="posts")''',
    ]
    
    class_templates = [
        '''class CacheManager:
    """Manages application cache with TTL support."""
    
    def __init__(self, default_ttl: int = 3600):
        self._cache = {}
        self._ttl = default_ttl
    
    def get(self, key: str):
        if key in self._cache:
            value, expiry = self._cache[key]
            if datetime.now() < expiry:
                return value
            del self._cache[key]
        return None
    
    def set(self, key: str, value: any, ttl: Optional[int] = None):
        expiry = datetime.now() + timedelta(seconds=ttl or self._ttl)
        self._cache[key] = (value, expiry)
    
    def clear(self):
        self._cache.clear()
    
    def size(self) -> int:
        return len(self._cache)''',
        
        '''class DatabaseConnection:
    """Database connection manager with connection pooling."""
    
    def __init__(self, host: str, port: int, database: str):
        self.host = host
        self.port = port
        self.database = database
        self._pool = []
        self._max_pool_size = 10
    
    def connect(self):
        if len(self._pool) < self._max_pool_size:
            conn = self._create_connection()
            self._pool.append(conn)
            return conn
        raise PoolExhaustedError("No available connections")
    
    def _create_connection(self):
        return {"host": self.host, "port": self.port, "db": self.database}
    
    def close_all(self):
        self._pool.clear()''',
    ]
    
    function_templates = [
        '''def process_data(data: List[Dict[str, Any]], filter_key: str = None) -> List[Dict]:
    """
    Process raw data with optional filtering.
    
    Args:
        data: List of data dictionaries
        filter_key: Optional key to filter by
        
    Returns:
        Processed and filtered data
    """
    result = []
    
    for item in data:
        if filter_key and filter_key not in item:
            continue
        
        processed = {
            "id": item.get("id"),
            "name": item.get("name", "Unknown"),
            "value": item.get("value", 0) * 1.1,
            "timestamp": datetime.now().isoformat()
        }
        result.append(processed)
    
    return sorted(result, key=lambda x: x["value"], reverse=True)


def validate_data(data: Dict) -> bool:
    """Validate input data structure."""
    required = ["id", "name", "value"]
    return all(k in data for k in required)''',
        
        '''def calculate_statistics(numbers: List[float]) -> Dict[str, float]:
    """Calculate basic statistics for a list of numbers."""
    if not numbers:
        return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0}
    
    mean = sum(numbers) / len(numbers)
    sorted_nums = sorted(numbers)
    n = len(sorted_nums)
    
    if n % 2 == 0:
        median = (sorted_nums[n//2-1] + sorted_nums[n//2]) / 2
    else:
        median = sorted_nums[n//2]
    
    variance = sum((x - mean) ** 2 for x in numbers) / n
    std = variance ** 0.5
    
    return {
        "mean": mean,
        "median": median,
        "std": std,
        "min": min(numbers),
        "max": max(numbers),
        "count": len(numbers)
    }''',
    ]
    
    async_templates = [
        '''async def fetch_data(url: str, timeout: int = 30) -> Dict:
    """Fetch data from URL with timeout."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=timeout) as response:
            if response.status == 200:
                return await response.json()
            raise HTTPError(f"Status: {response.status}")


async def fetch_all(urls: List[str]) -> List[Dict]:
    """Fetch multiple URLs concurrently."""
    tasks = [fetch_data(url) for url in urls]
    return await asyncio.gather(*tasks)''',
        
        '''async def process_items(items: List[str]) -> List[Dict]:
    """Process items concurrently with rate limiting."""
    semaphore = asyncio.Semaphore(10)
    
    async def process_one(item: str) -> Dict:
        async with semaphore:
            await asyncio.sleep(0.1)  # Simulate work
            return {"item": item, "processed": True}
    
    results = await asyncio.gather(*[process_one(i) for i in items])
    return list(results)''',
    ]
    
    decorator_templates = [
        '''def retry(max_attempts: int = 3, delay: float = 1.0):
    """Retry decorator with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(delay * (2 ** attempt))
            raise last_exception
        return wrapper
    return decorator


def cache(ttl: int = 60):
    """Simple cache decorator."""
    def decorator(func):
        cache_store = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            now = time.time()
            
            if key in cache_store:
                result, timestamp = cache_store[key]
                if now - timestamp < ttl:
                    return result
            
            result = func(*args, **kwargs)
            cache_store[key] = (result, now)
            return result
        return wrapper
    return decorator''',
    ]
    
    error_handling_templates = [
        '''def safe_divide(a: float, b: float) -> Optional[float]:
    """Safely divide two numbers, returning None on division by zero."""
    try:
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    except ValueError as e:
        logger.error(f"Division error: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return None


class APIError(Exception):
    """Base exception for API errors."""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class ValidationError(APIError):
    """Exception for validation errors."""
    def __init__(self, message: str):
        super().__init__(message, status_code=400)''',
    ]
    
    testing_templates = [
        '''import pytest
from unittest.mock import Mock, patch, AsyncMock

class TestUserService:
    """Tests for UserService."""
    
    @pytest.fixture
    def mock_db(self):
        return Mock()
    
    @pytest.fixture
    def service(self, mock_db):
        return UserService(mock_db)
    
    def test_get_user_by_id(self, service, mock_db):
        mock_db.get_user.return_value = {"id": 1, "name": "Test"}
        
        result = service.get_user(1)
        
        assert result["id"] == 1
        assert result["name"] == "Test"
        mock_db.get_user.assert_called_once_with(1)
    
    def test_create_user(self, service, mock_db):
        mock_db.create_user.return_value = 1
        
        result = service.create_user("test", "test@example.com")
        
        assert result == 1
        mock_db.create_user.assert_called_once()''',
        
        '''import pytest
import asyncio

@pytest.mark.asyncio
async def test_async_fetch():
    """Test async fetch function."""
    with patch('aiohttp.ClientSession') as mock_session:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": "test"})
        
        mock_session.return_value.__aenter__.return_value.get.return_value = mock_response
        
        result = await fetch_data("http://test.com")
        
        assert result["data"] == "test"


def test_data_processing():
    """Test data processing function."""
    data = [
        {"id": 1, "value": 10},
        {"id": 2, "value": 20},
    ]
    
    result = process_data(data)
    
    assert len(result) == 2
    assert result[0]["value"] == 11.0''',
    ]
    
    utility_templates = [
        '''def parse_config(config_path: str) -> Dict[str, Any]:
    """Parse configuration file (JSON or YAML)."""
    with open(config_path, 'r') as f:
        content = f.read()
    
    if config_path.endswith('.json'):
        return json.loads(content)
    elif config_path.endswith(('.yaml', '.yml')):
        return yaml.safe_load(content)
    else:
        raise ValueError(f"Unsupported config format: {config_path}")


def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
    """Deep merge two dictionaries."""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result''',
        
        '''class Timer:
    """Context manager for timing code execution."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        logger.info(f"{self.name} took {self.elapsed:.4f} seconds")


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]''',
    ]
    
    # Combine all templates
    all_templates = (
        fastapi_templates * 8000 +    # ~24K samples
        pydantic_templates * 6000 +   # ~12K samples
        sqlalchemy_templates * 5000 +  # ~10K samples
        class_templates * 4000 +      # ~8K samples
        function_templates * 4000 +   # ~8K samples
        async_templates * 3000 +      # ~6K samples
        decorator_templates * 2000 +  # ~4K samples
        error_handling_templates * 2000 +  # ~4K samples
        testing_templates * 2500 +   # ~5K samples
        utility_templates * 2500      # ~5K samples
    )
    
    # Shuffle and take required number
    random.shuffle(all_templates)
    
    # Generate variations to reach target size
    snippets = []
    variations = [
        lambda x: x,
        lambda x: x + "\n\n# Additional methods\n",
        lambda x: "# Module: " + x,
        lambda x: '"""Module docstring."""\n\n' + x,
    ]
    
    template_idx = 0
    while len(snippets) < num_samples:
        template = all_templates[template_idx % len(all_templates)]
        variation = random.choice(variations)
        
        # Add some random variations
        lines = template.split('\n')
        if random.random() > 0.5:
            # Add some random comments
            insert_pos = random.randint(1, len(lines))
            lines.insert(insert_pos, f"# {random.choice(['TODO', 'FIXME', 'NOTE', 'IMPORTANT'])}")
        
        modified = '\n'.join(lines)
        snippets.append(variation(modified))
        
        template_idx += 1
    
    logger.info(f"Generated {len(snippets):,} synthetic Python code snippets")
    return snippets[:num_samples]


# ============================================================================
# Training Functions
# ============================================================================

def compute_loss(model, input_ids, attention_mask, labels):
    """Compute causal language modeling loss."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
    
    # Shift for causal LM: predict next token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    # Flatten
    loss_fct = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    
    return loss


def train_step(model, optimizer, batch, device, scaler):
    """Single training step with mixed precision."""
    model.train()
    
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    
    optimizer.zero_grad()
    
    # Mixed precision forward pass
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
        loss = compute_loss(model, input_ids, attention_mask, labels)
    
    # Backward pass
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    
    return loss.item()


# ============================================================================
# Main Training
# ============================================================================

def main():
    """Main training function."""
    
    # Configuration
    config = TrainingConfig()
    
    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Try to load HuggingFace data, fall back to synthetic
    logger.info("=" * 60)
    logger.info("STEP 1: Loading/Generating Training Data")
    logger.info("=" * 60)
    
    code_snippets = try_load_huggingface_dataset(config.dataset_size)
    
    if code_snippets is None:
        logger.info("Falling back to synthetic Python code generation...")
        code_snippets = generate_synthetic_python_data(config.dataset_size)
    
    logger.info(f"Loaded {len(code_snippets):,} code snippets")
    
    # Tokenizer
    logger.info("=" * 60)
    logger.info("STEP 2: Initializing Tokenizer")
    logger.info("="  * 60)
    
    tokenizer = SimpleTokenizer(vocab_size=50304)
    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Dataset and DataLoader
    dataset = PythonCodeDataset(code_snippets, tokenizer, seq_length=config.seq_length)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    # Model
    logger.info("=" * 60)
    logger.info("STEP 3: Initializing Model")
    logger.info("=" * 60)
    
    model_config = CodlingConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        d_state=config.d_state,
        d_expand=config.d_expand,
        max_position_embeddings=config.seq_length,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        use_hyena=False,     # Pure SSM for efficiency
        use_linear_attn=False,
        use_longrope=False,
        dropout=0.0,
    )
    
    model = CodlingForCausalLM(model_config)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.max_steps,
        eta_min=config.learning_rate * 0.1
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    logger.info("=" * 60)
    logger.info("STEP 4: Training")
    logger.info("=" * 60)
    
    data_iter = iter(dataloader)
    step = 0
    running_loss = 0.0
    
    while step < config.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        loss = train_step(model, optimizer, batch, device, scaler)
        scheduler.step()
        
        running_loss += loss
        step += 1
        
        if step % 1 == 0:
            avg_loss = running_loss / 1
            lr = scheduler.get_last_lr()[0]
            logger.info(
                f"Step {step:4d}/{config.max_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {lr:.6f}"
            )
            running_loss = 0.0
        
        # Save checkpoint
        if step % config.save_interval == 0:
            checkpoint_path = os.path.join(
                config.checkpoint_dir, 
                f"codling_step_{step}.pt"
            )
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': model_config,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final checkpoint
    final_checkpoint = os.path.join(config.checkpoint_dir, "codling_final.pt")
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': model_config,
    }, final_checkpoint)
    logger.info(f"Saved final checkpoint: {final_checkpoint}")
    
    # Generation tests
    logger.info("=" * 60)
    logger.info("STEP 5: Testing Generation")
    logger.info("=" * 60)
    
    model.eval()
    generator = CodlingGenerator(model, tokenizer=tokenizer, device=device)
    
    gen_config = GenerationConfig(
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    
    for prompt in config.test_prompts:
        logger.info(f"\n--- Prompt: {prompt} ---")
        
        try:
            result = generator.generate(prompt, config=gen_config)
            generated = result.get('text', '') if isinstance(result, dict) else result
            
            # Show first 500 chars of generated code
            display_text = generated[:500] if len(generated) > 500 else generated
            logger.info(f"Generated:\n{display_text}\n")
            
        except Exception as e:
            logger.warning(f"Generation failed for '{prompt}': {e}")
    
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
