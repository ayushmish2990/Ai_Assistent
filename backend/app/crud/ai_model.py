from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session

from app import models, schemas
from app.crud.base import CRUDBase

class CRUDAIModel(CRUDBase[models.AIModel, schemas.AIModelCreate, schemas.AIModelUpdate]):
    def get_multi_by_owner(
        self, db: Session, *, owner_id: int, skip: int = 0, limit: int = 100
    ) -> List[models.AIModel]:
        return (
            db.query(self.model)
            .filter(models.AIModel.owner_id == owner_id)
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def update_status(
        self, db: Session, *, db_obj: models.AIModel, status: str
    ) -> models.AIModel:
        db_obj.status = status
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj
    
    def update_metrics(
        self, db: Session, *, db_obj: models.AIModel, metrics: Dict[str, Any]
    ) -> models.AIModel:
        db_obj.metrics = metrics
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

ai_model = CRUDAIModel(models.AIModel)
