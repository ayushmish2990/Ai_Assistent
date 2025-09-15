from sqlalchemy import Column, Integer, String, Text, ForeignKey, Enum, JSON, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.db.base_class import Base
import enum

class ModelStatus(str, enum.Enum):
    TRAINING = "training"
    READY = "ready"
    FAILED = "failed"
    OFFLINE = "offline"

class AIModel(Base):
    """AI Model for storing model information"""
    __tablename__ = "ai_models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    description = Column(Text, nullable=True)
    model_path = Column(String, nullable=False)
    status = Column(Enum(ModelStatus), default=ModelStatus.OFFLINE)
    parameters = Column(JSON, default={})
    metrics = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Foreign Keys
    owner_id = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    owner = relationship("User", back_populates="ai_models")
    
    def __repr__(self):
        return f"<AIModel {self.name} ({self.status})>"
