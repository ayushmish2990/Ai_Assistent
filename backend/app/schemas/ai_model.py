from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class ModelStatus(str, Enum):
    TRAINING = "training"
    READY = "ready"
    FAILED = "failed"
    OFFLINE = "offline"

# Shared properties
class AIModelBase(BaseModel):
    name: str
    description: Optional[str] = None
    status: Optional[ModelStatus] = ModelStatus.OFFLINE
    parameters: Optional[Dict[str, Any]] = {}
    metrics: Optional[Dict[str, Any]] = {}

# Properties to receive on model creation
class AIModelCreate(AIModelBase):
    model_path: str
    owner_id: int

# Properties to receive on model update
class AIModelUpdate(AIModelBase):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[ModelStatus] = None
    parameters: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None

# Properties shared by models stored in DB
class AIModelInDBBase(AIModelBase):
    id: int
    owner_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

# Properties to return to client
class AIModel(AIModelInDBBase):
    pass

# Properties properties stored in DB
class AIModelInDB(AIModelInDBBase):
    model_path: str
