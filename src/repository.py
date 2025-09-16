"""
IRAM Repository Layer

Repository pattern implementation for data access with async SQLAlchemy,
CRUD operations, complex queries, and connection pooling.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Type, List, Optional, Dict, Any, Union
from sqlalchemy import select, update, delete, func, and_, or_, desc, asc
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from datetime import datetime, timedelta

from .db import session_scope
from .models import (
    Base, User, Job, InstagramProfile, InstagramPost, Analysis,
    AnalysisArtifact, ProfileMetric, UsageMetric, AuditLog, ApiKey,
    JobStatus, UserRole, AnalysisType, ScrapingMethod
)
from .logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar('T', bound=Base)


class BaseRepository(Generic[T], ABC):
    """Base repository with common CRUD operations."""
    
    def __init__(self, model: Type[T]):
        self.model = model
    
    async def create(self, **kwargs) -> T:
        """Create a new entity."""
        try:
            async with session_scope() as session:
                entity = self.model(**kwargs)
                session.add(entity)
                await session.flush()
                await session.refresh(entity)
                return entity
        except IntegrityError as e:
            logger.error(f"Integrity error creating {self.model.__name__}: {e}")
            raise
        except SQLAlchemyError as e:
            logger.error(f"Database error creating {self.model.__name__}: {e}")
            raise
    
    async def get_by_id(self, entity_id: Union[int, str]) -> Optional[T]:
        """Get entity by ID."""
        try:
            async with session_scope() as session:
                return await session.get(self.model, entity_id)
        except SQLAlchemyError as e:
            logger.error(f"Database error getting {self.model.__name__} by ID {entity_id}: {e}")
            raise
    
    async def get_all(self, limit: int = 100, offset: int = 0) -> List[T]:
        """Get all entities with pagination."""
        try:
            async with session_scope() as session:
                result = await session.execute(
                    select(self.model)
                    .limit(limit)
                    .offset(offset)
                )
                return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error(f"Database error getting all {self.model.__name__}: {e}")
            raise
    
    async def update(self, entity_id: Union[int, str], **kwargs) -> Optional[T]:
        """Update entity by ID."""
        try:
            async with session_scope() as session:
                entity = await session.get(self.model, entity_id)
                if entity:
                    for key, value in kwargs.items():
                        setattr(entity, key, value)
                    await session.flush()
                    await session.refresh(entity)
                return entity
        except SQLAlchemyError as e:
            logger.error(f"Database error updating {self.model.__name__} {entity_id}: {e}")
            raise
    
    async def delete(self, entity_id: Union[int, str]) -> bool:
        """Delete entity by ID."""
        try:
            async with session_scope() as session:
                entity = await session.get(self.model, entity_id)
                if entity:
                    await session.delete(entity)
                    await session.flush()
                    return True
                return False
        except SQLAlchemyError as e:
            logger.error(f"Database error deleting {self.model.__name__} {entity_id}: {e}")
            raise
    
    async def count(self, **filters) -> int:
        """Count entities with optional filters."""
        try:
            async with session_scope() as session:
                query = select(func.count(self.model.id))
                if filters:
                    conditions = []
                    for key, value in filters.items():
                        if hasattr(self.model, key):
                            conditions.append(getattr(self.model, key) == value)
                    if conditions:
                        query = query.where(and_(*conditions))
                
                result = await session.execute(query)
                return result.scalar() or 0
        except SQLAlchemyError as e:
            logger.error(f"Database error counting {self.model.__name__}: {e}")
            raise


class UserRepository(BaseRepository[User]):
    """Repository for user operations."""
    
    def __init__(self):
        super().__init__(User)
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email address."""
        try:
            async with session_scope() as session:
                result = await session.execute(
                    select(User).where(User.email == email.lower())
                )
                return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            logger.error(f"Database error getting user by email: {e}")
            raise
    
    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        try:
            async with session_scope() as session:
                result = await session.execute(
                    select(User).where(User.username == username.lower())
                )
                return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            logger.error(f"Database error getting user by username: {e}")
            raise
    
    async def get_by_external_id(self, provider: str, external_id: str) -> Optional[User]:
        """Get user by external auth provider ID."""
        try:
            async with session_scope() as session:
                result = await session.execute(
                    select(User).where(
                        and_(
                            User.auth_provider == provider,
                            User.external_id == external_id
                        )
                    )
                )
                return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            logger.error(f"Database error getting user by external ID: {e}")
            raise
    
    async def get_active_users(self, role: Optional[UserRole] = None) -> List[User]:
        """Get all active users, optionally filtered by role."""
        try:
            async with session_scope() as session:
                query = select(User).where(User.is_active == True)
                if role:
                    query = query.where(User.role == role)
                
                result = await session.execute(query.order_by(User.created_at.desc()))
                return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error(f"Database error getting active users: {e}")
            raise
    
    async def update_last_login(self, user_id: int) -> None:
        """Update user's last login timestamp."""
        await self.update(user_id, last_login_at=datetime.utcnow())
    
    async def increment_usage_counters(self, user_id: int, jobs: int = 0, analyses: int = 0) -> None:
        """Increment user's usage counters."""
        try:
            async with session_scope() as session:
                user = await session.get(User, user_id)
                if user:
                    user.total_jobs += jobs
                    user.total_analyses += analyses
                    await session.flush()
        except SQLAlchemyError as e:
            logger.error(f"Database error updating usage counters for user {user_id}: {e}")
            raise


class JobRepository(BaseRepository[Job]):
    """Repository for job operations."""
    
    def __init__(self):
        super().__init__(Job)
    
    async def get_by_status(
        self, 
        status: JobStatus, 
        limit: int = 100, 
        user_id: Optional[int] = None
    ) -> List[Job]:
        """Get jobs by status."""
        try:
            async with session_scope() as session:
                query = select(Job).where(Job.status == status)
                
                if user_id:
                    query = query.where(Job.user_id == user_id)
                
                query = query.order_by(Job.priority.desc(), Job.created_at.asc()).limit(limit)
                
                result = await session.execute(query)
                return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error(f"Database error getting jobs by status: {e}")
            raise
    
    async def get_queued_jobs(self, limit: int = 50) -> List[Job]:
        """Get queued jobs ordered by priority and creation time."""
        return await self.get_by_status(JobStatus.queued, limit)
    
    async def get_scheduled_jobs(self, before: Optional[datetime] = None) -> List[Job]:
        """Get jobs scheduled to run before the specified time."""
        if before is None:
            before = datetime.utcnow()
        
        try:
            async with session_scope() as session:
                result = await session.execute(
                    select(Job).where(
                        and_(
                            Job.status == JobStatus.queued,
                            Job.scheduled_for <= before
                        )
                    ).order_by(Job.scheduled_for.asc())
                )
                return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error(f"Database error getting scheduled jobs: {e}")
            raise
    
    async def get_user_jobs(
        self, 
        user_id: int, 
        limit: int = 50, 
        offset: int = 0,
        status: Optional[JobStatus] = None
    ) -> List[Job]:
        """Get jobs for a specific user."""
        try:
            async with session_scope() as session:
                query = select(Job).where(Job.user_id == user_id)
                
                if status:
                    query = query.where(Job.status == status)
                
                query = query.order_by(Job.created_at.desc()).limit(limit).offset(offset)
                
                result = await session.execute(query)
                return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error(f"Database error getting user jobs: {e}")
            raise
    
    async def update_status(
        self, 
        job_id: int, 
        status: JobStatus, 
        progress: Optional[int] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None
    ) -> Optional[Job]:
        """Update job status and related fields."""
        update_data = {"status": status, "updated_at": datetime.utcnow()}
        
        if progress is not None:
            update_data["progress"] = progress
        if result is not None:
            update_data["result"] = result
        if error is not None:
            update_data["error"] = error
        
        if status == JobStatus.running:
            update_data["started_at"] = datetime.utcnow()
        elif status in [JobStatus.completed, JobStatus.failed, JobStatus.cancelled]:
            update_data["completed_at"] = datetime.utcnow()
        
        return await self.update(job_id, **update_data)
    
    async def get_job_statistics(
        self, 
        user_id: Optional[int] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get job statistics for the specified period."""
        since_date = datetime.utcnow() - timedelta(days=days)
        
        try:
            async with session_scope() as session:
                base_query = select(Job).where(Job.created_at >= since_date)
                if user_id:
                    base_query = base_query.where(Job.user_id == user_id)
                
                # Count by status
                status_counts = {}
                for status in JobStatus:
                    count_query = select(func.count(Job.id)).where(
                        and_(
                            Job.created_at >= since_date,
                            Job.status == status
                        )
                    )
                    if user_id:
                        count_query = count_query.where(Job.user_id == user_id)
                    
                    result = await session.execute(count_query)
                    status_counts[status.value] = result.scalar() or 0
                
                # Average execution time for completed jobs
                avg_time_query = select(func.avg(Job.execution_time_ms)).where(
                    and_(
                        Job.created_at >= since_date,
                        Job.status == JobStatus.completed,
                        Job.execution_time_ms.is_not(None)
                    )
                )
                if user_id:
                    avg_time_query = avg_time_query.where(Job.user_id == user_id)
                
                avg_time_result = await session.execute(avg_time_query)
                avg_execution_time = avg_time_result.scalar()
                
                return {
                    "period_days": days,
                    "status_counts": status_counts,
                    "total_jobs": sum(status_counts.values()),
                    "success_rate": status_counts.get("completed", 0) / max(1, sum(status_counts.values())),
                    "avg_execution_time_ms": avg_execution_time,
                }
        except SQLAlchemyError as e:
            logger.error(f"Database error getting job statistics: {e}")
            raise


class InstagramProfileRepository(BaseRepository[InstagramProfile]):
    """Repository for Instagram profile operations."""
    
    def __init__(self):
        super().__init__(InstagramProfile)
    
    async def get_by_username(self, username: str) -> Optional[InstagramProfile]:
        """Get profile by username."""
        try:
            async with session_scope() as session:
                result = await session.execute(
                    select(InstagramProfile).where(
                        InstagramProfile.username == username.lower()
                    )
                )
                return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            logger.error(f"Database error getting profile by username: {e}")
            raise
    
    async def get_with_posts(
        self, 
        username: str, 
        post_limit: int = 50
    ) -> Optional[InstagramProfile]:
        """Get profile with recent posts."""
        try:
            async with session_scope() as session:
                result = await session.execute(
                    select(InstagramProfile)
                    .options(
                        selectinload(InstagramProfile.posts)
                        .options(
                            lambda q: q.order_by(InstagramPost.posted_at.desc())
                            .limit(post_limit)
                        )
                    )
                    .where(InstagramProfile.username == username.lower())
                )
                return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            logger.error(f"Database error getting profile with posts: {e}")
            raise
    
    async def get_profiles_by_metrics(
        self,
        min_followers: Optional[int] = None,
        max_followers: Optional[int] = None,
        is_verified: Optional[bool] = None,
        is_business: Optional[bool] = None,
        limit: int = 100
    ) -> List[InstagramProfile]:
        """Get profiles filtered by metrics and status."""
        try:
            async with session_scope() as session:
                query = select(InstagramProfile)
                conditions = []
                
                if min_followers is not None:
                    conditions.append(InstagramProfile.followers >= min_followers)
                if max_followers is not None:
                    conditions.append(InstagramProfile.followers <= max_followers)
                if is_verified is not None:
                    conditions.append(InstagramProfile.is_verified == is_verified)
                if is_business is not None:
                    conditions.append(InstagramProfile.is_business == is_business)
                
                if conditions:
                    query = query.where(and_(*conditions))
                
                query = query.order_by(InstagramProfile.followers.desc()).limit(limit)
                
                result = await session.execute(query)
                return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error(f"Database error getting profiles by metrics: {e}")
            raise
    
    async def get_stale_profiles(self, hours: int = 24) -> List[InstagramProfile]:
        """Get profiles that haven't been scraped recently."""
        stale_time = datetime.utcnow() - timedelta(hours=hours)
        
        try:
            async with session_scope() as session:
                result = await session.execute(
                    select(InstagramProfile).where(
                        or_(
                            InstagramProfile.last_scraped_at.is_(None),
                            InstagramProfile.last_scraped_at <= stale_time
                        )
                    ).order_by(InstagramProfile.followers.desc())
                )
                return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error(f"Database error getting stale profiles: {e}")
            raise
    
    async def update_metrics(
        self,
        profile_id: int,
        followers: int,
        following: int,
        posts_count: int,
        scraping_method: ScrapingMethod
    ) -> None:
        """Update profile metrics and scraping timestamp."""
        await self.update(
            profile_id,
            followers=followers,
            following=following,
            posts_count=posts_count,
            last_scraped_at=datetime.utcnow(),
            scraping_method=scraping_method
        )


class AnalysisRepository(BaseRepository[Analysis]):
    """Repository for analysis operations."""
    
    def __init__(self):
        super().__init__(Analysis)
    
    async def get_by_type(
        self,
        analysis_type: AnalysisType,
        user_id: Optional[int] = None,
        limit: int = 50
    ) -> List[Analysis]:
        """Get analyses by type."""
        try:
            async with session_scope() as session:
                query = select(Analysis).where(Analysis.analysis_type == analysis_type)
                
                if user_id:
                    query = query.where(Analysis.user_id == user_id)
                
                query = query.order_by(Analysis.created_at.desc()).limit(limit)
                
                result = await session.execute(query)
                return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error(f"Database error getting analyses by type: {e}")
            raise
    
    async def get_profile_analyses(
        self,
        profile_id: int,
        analysis_type: Optional[AnalysisType] = None,
        limit: int = 50
    ) -> List[Analysis]:
        """Get analyses for a specific profile."""
        try:
            async with session_scope() as session:
                query = select(Analysis).where(Analysis.profile_id == profile_id)
                
                if analysis_type:
                    query = query.where(Analysis.analysis_type == analysis_type)
                
                query = query.order_by(Analysis.created_at.desc()).limit(limit)
                
                result = await session.execute(query)
                return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error(f"Database error getting profile analyses: {e}")
            raise
    
    async def get_with_artifacts(self, analysis_id: int) -> Optional[Analysis]:
        """Get analysis with all associated artifacts."""
        try:
            async with session_scope() as session:
                result = await session.execute(
                    select(Analysis)
                    .options(selectinload(Analysis.artifacts))
                    .where(Analysis.id == analysis_id)
                )
                return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            logger.error(f"Database error getting analysis with artifacts: {e}")
            raise


class AuditLogRepository(BaseRepository[AuditLog]):
    """Repository for audit log operations."""
    
    def __init__(self):
        super().__init__(AuditLog)
    
    async def log_action(
        self,
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        user_id: Optional[int] = None,
        api_key_id: Optional[int] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        old_values: Optional[Dict[str, Any]] = None,
        new_values: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        method: Optional[str] = None
    ) -> AuditLog:
        """Create an audit log entry."""
        return await self.create(
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            user_id=user_id,
            api_key_id=api_key_id,
            ip_address=ip_address,
            user_agent=user_agent,
            old_values=old_values,
            new_values=new_values,
            metadata=metadata,
            request_id=request_id,
            endpoint=endpoint,
            method=method
        )
    
    async def get_user_actions(
        self,
        user_id: int,
        days: int = 30,
        limit: int = 100
    ) -> List[AuditLog]:
        """Get audit logs for a specific user."""
        since_date = datetime.utcnow() - timedelta(days=days)
        
        try:
            async with session_scope() as session:
                result = await session.execute(
                    select(AuditLog)
                    .where(
                        and_(
                            AuditLog.user_id == user_id,
                            AuditLog.created_at >= since_date
                        )
                    )
                    .order_by(AuditLog.created_at.desc())
                    .limit(limit)
                )
                return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error(f"Database error getting user audit logs: {e}")
            raise
    
    async def get_resource_history(
        self,
        resource_type: str,
        resource_id: str,
        limit: int = 50
    ) -> List[AuditLog]:
        """Get audit history for a specific resource."""
        try:
            async with session_scope() as session:
                result = await session.execute(
                    select(AuditLog)
                    .where(
                        and_(
                            AuditLog.resource_type == resource_type,
                            AuditLog.resource_id == resource_id
                        )
                    )
                    .order_by(AuditLog.created_at.desc())
                    .limit(limit)
                )
                return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error(f"Database error getting resource audit history: {e}")
            raise


# Repository factory for dependency injection
class RepositoryFactory:
    """Factory for creating repository instances."""
    
    _repositories = {
        'user': UserRepository,
        'job': JobRepository,
        'profile': InstagramProfileRepository,
        'analysis': AnalysisRepository,
        'audit': AuditLogRepository,
    }
    
    @classmethod
    def get_repository(cls, name: str) -> BaseRepository:
        """Get repository instance by name."""
        if name not in cls._repositories:
            raise ValueError(f"Unknown repository: {name}")
        
        return cls._repositories[name]()
    
    @classmethod
    def register_repository(cls, name: str, repository_class: Type[BaseRepository]):
        """Register a new repository type."""
        cls._repositories[name] = repository_class


# Convenience functions for common operations
async def get_or_create_profile(username: str, **defaults) -> tuple[InstagramProfile, bool]:
    """Get existing profile or create a new one."""
    repo = RepositoryFactory.get_repository('profile')
    profile = await repo.get_by_username(username)
    
    if profile:
        return profile, False
    
    profile = await repo.create(username=username, **defaults)
    return profile, True


async def log_user_action(
    user_id: int,
    action: str,
    resource_type: str,
    resource_id: Optional[str] = None,
    **kwargs
) -> AuditLog:
    """Convenience function to log user actions."""
    audit_repo = RepositoryFactory.get_repository('audit')
    return await audit_repo.log_action(
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        user_id=user_id,
        **kwargs
    )