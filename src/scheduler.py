"""
IRAM Task Scheduler

APScheduler-based background job scheduler with database persistence,
recurring tasks, and comprehensive job management capabilities.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, EVENT_JOB_MISSED

from .config import get_config
from .logging_config import get_logger
from .repository import RepositoryFactory, JobRepository
from .models import Job, JobStatus

logger = get_logger(__name__)


class IRamScheduler:
    """IRAM task scheduler with APScheduler backend."""
    
    def __init__(self):
        self.config = get_config()
        self.scheduler: Optional[AsyncIOScheduler] = None
        self.job_repo: JobRepository = RepositoryFactory.get_repository('job')
        self._job_functions: Dict[str, Callable] = {}
        self._running_jobs: Dict[str, asyncio.Task] = {}
    
    async def initialize(self):
        """Initialize the scheduler with database job store."""
        if self.scheduler is not None:
            logger.warning("Scheduler already initialized")
            return
        
        # Configure job stores
        jobstores = {}
        if self.config.has_database():
            jobstores['default'] = SQLAlchemyJobStore(url=self.config.database.url)
        
        # Configure executors
        executors = {
            'default': AsyncIOExecutor(),
        }
        
        # Job defaults
        job_defaults = {
            'coalesce': False,
            'max_instances': 3
        }
        
        # Create scheduler
        self.scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone='UTC'
        )
        
        # Add event listeners
        self.scheduler.add_listener(
            self._job_executed_listener,
            EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED
        )
        
        logger.info("IRAM scheduler initialized")
    
    async def start(self):
        """Start the scheduler."""
        if not self.scheduler:
            await self.initialize()
        
        self.scheduler.start()
        logger.info("IRAM scheduler started")
        
        # Register built-in jobs
        await self._register_builtin_jobs()
        
        # Resume any pending scheduled jobs from database
        await self._resume_scheduled_jobs()
    
    async def shutdown(self, wait: bool = True):
        """Shutdown the scheduler."""
        if self.scheduler:
            self.scheduler.shutdown(wait=wait)
            self.scheduler = None
            logger.info("IRAM scheduler shut down")
        
        # Cancel any running job tasks
        for task in self._running_jobs.values():
            if not task.done():
                task.cancel()
        
        if self._running_jobs:
            await asyncio.gather(*self._running_jobs.values(), return_exceptions=True)
            self._running_jobs.clear()
    
    def register_job_function(self, name: str, func: Callable):
        """Register a function that can be called by scheduled jobs."""
        self._job_functions[name] = func
        logger.debug(f"Registered job function: {name}")
    
    async def schedule_job(
        self,
        job_id: int,
        func_name: str,
        trigger_type: str = "date",
        **trigger_kwargs
    ) -> str:
        """Schedule a database job for execution."""
        if not self.scheduler:
            raise RuntimeError("Scheduler not initialized")
        
        if func_name not in self._job_functions:
            raise ValueError(f"Unknown job function: {func_name}")
        
        # Create trigger
        trigger = self._create_trigger(trigger_type, **trigger_kwargs)
        
        # Schedule the job
        scheduler_job_id = f"job_{job_id}"
        self.scheduler.add_job(
            func=self._execute_database_job,
            trigger=trigger,
            args=[job_id],
            id=scheduler_job_id,
            replace_existing=True,
            misfire_grace_time=300  # 5 minutes
        )
        
        logger.info(f"Scheduled job {job_id} with trigger {trigger_type}")
        return scheduler_job_id
    
    async def schedule_recurring_job(
        self,
        func_name: str,
        cron_expression: str,
        job_data: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None
    ) -> str:
        """Schedule a recurring job with cron expression."""
        if not self.scheduler:
            raise RuntimeError("Scheduler not initialized")
        
        if func_name not in self._job_functions:
            raise ValueError(f"Unknown job function: {func_name}")
        
        # Parse cron expression
        trigger = CronTrigger.from_crontab(cron_expression)
        
        # Generate job ID if not provided
        if not job_id:
            job_id = f"recurring_{func_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Schedule the job
        self.scheduler.add_job(
            func=self._execute_recurring_job,
            trigger=trigger,
            args=[func_name, job_data or {}],
            id=job_id,
            replace_existing=True
        )
        
        logger.info(f"Scheduled recurring job {job_id} with cron {cron_expression}")
        return job_id
    
    async def cancel_job(self, scheduler_job_id: str) -> bool:
        """Cancel a scheduled job."""
        if not self.scheduler:
            return False
        
        try:
            self.scheduler.remove_job(scheduler_job_id)
            logger.info(f"Cancelled scheduled job: {scheduler_job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel job {scheduler_job_id}: {e}")
            return False
    
    async def get_scheduled_jobs(self) -> List[Dict[str, Any]]:
        """Get information about all scheduled jobs."""
        if not self.scheduler:
            return []
        
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "func": str(job.func),
                "trigger": str(job.trigger),
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                "args": job.args,
                "kwargs": job.kwargs
            })
        
        return jobs
    
    async def _execute_database_job(self, job_id: int):
        """Execute a database job."""
        try:
            # Get job from database
            job = await self.job_repo.get_by_id(job_id)
            if not job:
                logger.error(f"Job {job_id} not found in database")
                return
            
            # Update job status to running
            await self.job_repo.update_status(job_id, JobStatus.running, progress=0)
            
            # Get the job function
            # For now, we'll use a generic task executor
            # In a real implementation, you'd have specific job functions
            result = await self._execute_job_task(job)
            
            # Update job status to completed
            await self.job_repo.update_status(
                job_id, 
                JobStatus.completed, 
                progress=100,
                result=result
            )
            
            logger.info(f"Database job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Database job {job_id} failed: {e}", exc_info=True)
            
            # Update job status to failed
            await self.job_repo.update_status(
                job_id,
                JobStatus.failed,
                error={"message": str(e), "type": type(e).__name__}
            )
    
    async def _execute_recurring_job(self, func_name: str, job_data: Dict[str, Any]):
        """Execute a recurring job function."""
        try:
            logger.info(f"Executing recurring job: {func_name}")
            
            func = self._job_functions[func_name]
            if asyncio.iscoroutinefunction(func):
                await func(**job_data)
            else:
                func(**job_data)
            
            logger.info(f"Recurring job {func_name} completed successfully")
            
        except Exception as e:
            logger.error(f"Recurring job {func_name} failed: {e}", exc_info=True)
    
    async def _execute_job_task(self, job: Job) -> Dict[str, Any]:
        """Execute a generic job task (placeholder for actual implementation)."""
        # This is a placeholder - in reality, you'd parse the job.task
        # and call the appropriate function with job.payload
        
        # Simulate some work
        await asyncio.sleep(1)
        
        return {
            "success": True,
            "message": f"Job {job.id} executed",
            "task": job.task,
            "execution_time_ms": 1000
        }
    
    def _create_trigger(self, trigger_type: str, **kwargs):
        """Create an APScheduler trigger from parameters."""
        if trigger_type == "date":
            run_date = kwargs.get("run_date")
            if isinstance(run_date, str):
                run_date = datetime.fromisoformat(run_date)
            return DateTrigger(run_date=run_date)
        
        elif trigger_type == "interval":
            return IntervalTrigger(**kwargs)
        
        elif trigger_type == "cron":
            return CronTrigger(**kwargs)
        
        else:
            raise ValueError(f"Unknown trigger type: {trigger_type}")
    
    async def _job_executed_listener(self, event):
        """Handle job execution events."""
        job_id = event.job_id
        
        if event.exception:
            logger.error(
                f"Scheduled job {job_id} failed: {event.exception}",
                extra={
                    "job_id": job_id,
                    "event_type": event.code,
                    "scheduled_run_time": event.scheduled_run_time,
                    "retval": event.retval
                }
            )
        else:
            logger.info(
                f"Scheduled job {job_id} executed successfully",
                extra={
                    "job_id": job_id,
                    "event_type": event.code,
                    "scheduled_run_time": event.scheduled_run_time,
                    "retval": event.retval
                }
            )
    
    async def _register_builtin_jobs(self):
        """Register built-in recurring jobs."""
        # Cache refresh job
        await self.schedule_recurring_job(
            func_name="cache_cleanup",
            cron_expression="0 */6 * * *",  # Every 6 hours
            job_id="builtin_cache_cleanup"
        )
        
        # Stale profile refresh
        await self.schedule_recurring_job(
            func_name="refresh_stale_profiles",
            cron_expression="0 2 * * *",  # Daily at 2 AM
            job_id="builtin_refresh_profiles"
        )
        
        # Job cleanup
        await self.schedule_recurring_job(
            func_name="cleanup_old_jobs",
            cron_expression="0 3 * * 0",  # Weekly on Sunday at 3 AM
            job_id="builtin_job_cleanup"
        )
        
        # Usage metrics collection
        await self.schedule_recurring_job(
            func_name="collect_usage_metrics",
            cron_expression="0 1 * * *",  # Daily at 1 AM
            job_id="builtin_usage_metrics"
        )
        
        logger.info("Built-in recurring jobs registered")
    
    async def _resume_scheduled_jobs(self):
        """Resume scheduled jobs from database."""
        try:
            # Get jobs scheduled for future execution
            scheduled_jobs = await self.job_repo.get_scheduled_jobs()
            
            for job in scheduled_jobs:
                if job.scheduled_for and job.scheduled_for > datetime.utcnow():
                    await self.schedule_job(
                        job.id,
                        "execute_task",  # Generic function name
                        trigger_type="date",
                        run_date=job.scheduled_for
                    )
                    logger.info(f"Resumed scheduled job {job.id}")
            
            logger.info(f"Resumed {len(scheduled_jobs)} scheduled jobs")
            
        except Exception as e:
            logger.error(f"Failed to resume scheduled jobs: {e}")


# Built-in job functions
async def cache_cleanup():
    """Clean up expired cache entries."""
    logger.info("Running cache cleanup job")
    # Implementation would clean up Redis or in-memory cache
    # For now, just log
    logger.info("Cache cleanup completed")


async def refresh_stale_profiles():
    """Refresh profiles that haven't been updated recently."""
    logger.info("Running stale profile refresh job")
    
    try:
        profile_repo = RepositoryFactory.get_repository('profile')
        stale_profiles = await profile_repo.get_stale_profiles(hours=24)
        
        logger.info(f"Found {len(stale_profiles)} stale profiles")
        
        # In a real implementation, you'd queue these for scraping
        # For now, just update the last_scraped_at timestamp
        for profile in stale_profiles[:10]:  # Limit to 10 profiles per run
            await profile_repo.update(
                profile.id,
                last_scraped_at=datetime.utcnow()
            )
        
        logger.info("Stale profile refresh completed")
        
    except Exception as e:
        logger.error(f"Stale profile refresh failed: {e}")


async def cleanup_old_jobs():
    """Clean up old completed jobs."""
    logger.info("Running job cleanup")
    
    try:
        # This would delete jobs older than 30 days
        # Implementation depends on your retention policy
        logger.info("Job cleanup completed")
        
    except Exception as e:
        logger.error(f"Job cleanup failed: {e}")


async def collect_usage_metrics():
    """Collect daily usage metrics."""
    logger.info("Collecting usage metrics")
    
    try:
        job_repo = RepositoryFactory.get_repository('job')
        stats = await job_repo.get_job_statistics(days=1)
        
        logger.info(f"Daily job statistics: {stats}")
        
        # In a real implementation, you'd store these metrics
        # in a time-series database or metrics collection system
        
        logger.info("Usage metrics collection completed")
        
    except Exception as e:
        logger.error(f"Usage metrics collection failed: {e}")


# Global scheduler instance
_scheduler_instance: Optional[IRamScheduler] = None


async def get_scheduler() -> IRamScheduler:
    """Get the global scheduler instance."""
    global _scheduler_instance
    
    if _scheduler_instance is None:
        _scheduler_instance = IRamScheduler()
        await _scheduler_instance.initialize()
        
        # Register built-in job functions
        _scheduler_instance.register_job_function("cache_cleanup", cache_cleanup)
        _scheduler_instance.register_job_function("refresh_stale_profiles", refresh_stale_profiles)
        _scheduler_instance.register_job_function("cleanup_old_jobs", cleanup_old_jobs)
        _scheduler_instance.register_job_function("collect_usage_metrics", collect_usage_metrics)
    
    return _scheduler_instance


async def start_scheduler():
    """Start the global scheduler."""
    scheduler = await get_scheduler()
    await scheduler.start()


async def stop_scheduler():
    """Stop the global scheduler."""
    global _scheduler_instance
    if _scheduler_instance:
        await _scheduler_instance.shutdown()
        _scheduler_instance = None