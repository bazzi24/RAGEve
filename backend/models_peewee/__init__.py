"""
Peewee ORM models with RAGEve schema.

This module provides:
- Database connection management (singleton with retry logic)
- All 27 database models
- init_db() to create all tables
- close_db() for cleanup
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import peewee
from playhouse.pool import PooledMySQLDatabase

from backend.config_loader import settings

# Set up logger
_log = logging.getLogger(__name__)

from .base import BaseModel  # noqa: E402
from .canvas import CanvasTemplate, UserCanvas  # noqa: E402
from .connector import Connector, Connector2Kb, SyncLogs  # noqa: E402
from .dialog import Conversation, Dialog  # noqa: E402
from .evaluation import (  # noqa: E402
    EvaluationCase,
    EvaluationDataset,
    EvaluationResult,
    EvaluationRun,
)
from .knowledgebase import (  # noqa: E402
    Document,
    File,
    File2Document,
    Knowledgebase,
    Task,
)
from .llm import LLM, LLMFactories, TenantLLM  # noqa: E402
from .system import (  # noqa: E402
    MCP,
    API4Conversation,
    APIToken,
    PipelineOperationLog,
    Search,
    SystemSettings,
)
from .user import Tenant, User, UserTenant  # noqa: E402

# ==================== Database Configuration ====================

MYSQL_CONFIG = {
    "host": settings.mysql_host,
    "port": int(settings.mysql_port),
    "user": settings.mysql_user,
    "password": settings.mysql_password,
    "database": settings.mysql_dbname,
    "charset": "utf8mb4",
    "sql_mode": "STRICT_TRANS_TABLES",
}

POOL_CONFIG = {
    "max_connections": 900,
    "stale_timeout": 300,
    "max_retries": 5,
    "retry_delay": 1,
}


class RetryingPooledMySQLDatabase(PooledMySQLDatabase):
    """Database with retry logic for connection failures."""

    def __init__(self, *args, max_retries=5, retry_delay=1, **kwargs):
        """
        Initialize the database with retry configuration.

        Args:
            max_retries: Maximum number of retry attempts for transient errors
            retry_delay: Base delay in seconds between retries (exponential backoff)
        """
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def execute_sql(self, *args, **kwargs):
        """Execute SQL with retry logic for transient connection errors."""
        for attempt in range(self.max_retries + 1):
            try:
                return super().execute_sql(*args, **kwargs)
            except (peewee.OperationalError, peewee.InterfaceError) as e:
                should_retry = (
                    (hasattr(e, "args") and e.args and e.args[0] in [2013, 2006])
                    or (str(e) in ["", "Lost connection"])
                    or (
                        hasattr(e, "__class__")
                        and e.__class__.__name__ == "InterfaceError"
                    )
                )
                if should_retry and attempt < self.max_retries:
                    _log.warning(
                        "Database connection issue (attempt %d/%d): %s",
                        attempt + 1,
                        self.max_retries,
                        e,
                    )
                    time.sleep(self.retry_delay * (2**attempt))
                    try:
                        self.close()
                        self.connect()
                    except Exception:
                        pass
                else:
                    _log.error("Database execution failed: %s", e)
                    raise
        return None


# Global database singleton
_database: Optional[RetryingPooledMySQLDatabase] = None


def get_database() -> RetryingPooledMySQLDatabase:
    """Get or create the database singleton."""
    global _database
    if _database is None:
        _database = _init_database()
    return _database


def _init_database() -> RetryingPooledMySQLDatabase:
    """Initialize database connection and create database if needed."""
    db_config = MYSQL_CONFIG.copy()
    db_name = db_config.pop("database")

    # Add pool config
    db_config.update(POOL_CONFIG)

    # Create database if it doesn't exist
    try:
        import pymysql

        conn = pymysql.connect(
            host=db_config["host"],
            port=db_config["port"],
            user=db_config["user"],
            password=db_config["password"],
            charset="utf8mb4",
        )
        with conn.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            _log.info("Database '%s' created or already exists", db_name)
        conn.close()
    except Exception as e:
        _log.error("Failed to create database: %s", e)
        raise

    # Connect to the database
    db_config["database"] = db_name
    database = RetryingPooledMySQLDatabase(**db_config)

    # Bind all models to this database
    bind_models(database)

    return database


def bind_models(database: peewee.Database):
    """Bind all model classes to the given database."""
    models = [
        User,
        Tenant,
        UserTenant,
        Knowledgebase,
        Document,
        File,
        File2Document,
        Task,
        Dialog,
        Conversation,
        LLMFactories,
        LLM,
        TenantLLM,
        Connector,
        Connector2Kb,
        SyncLogs,
        UserCanvas,
        CanvasTemplate,
        EvaluationDataset,
        EvaluationCase,
        EvaluationRun,
        EvaluationResult,
        SystemSettings,
        APIToken,
        API4Conversation,
        MCP,
        Search,
        PipelineOperationLog,
    ]
    for model in models:
        model._meta.database = database


async def init_db() -> None:
    """Create all tables if they don't exist. Async-compatible wrapper."""
    database = get_database()
    # Peewee is synchronous, but we can run in thread pool if needed
    # For now, direct call (FastAPI lifespan can handle sync init)
    with database.connection_context():
        database.create_tables(
            [
                User,
                Tenant,
                UserTenant,
                Knowledgebase,
                Document,
                File,
                File2Document,
                Task,
                Dialog,
                Conversation,
                LLMFactories,
                LLM,
                TenantLLM,
                Connector,
                Connector2Kb,
                SyncLogs,
                UserCanvas,
                CanvasTemplate,
                EvaluationDataset,
                EvaluationCase,
                EvaluationRun,
                EvaluationResult,
                SystemSettings,
                APIToken,
                API4Conversation,
                MCP,
                Search,
                PipelineOperationLog,
            ],
            safe=True,
        )
    _log.info("All tables initialized")


def close_db() -> None:
    """Close database connection pool."""
    global _database
    if _database is not None:
        _database.close()
        _database = None
        _log.info("Database connection pool closed")


# ==================== Export All Models ====================

__all__ = [
    # Base
    "BaseModel",
    "JSONTextField",
    "ListField",
    "get_database",
    "init_db",
    "close_db",
    # User & Tenancy
    "User",
    "Tenant",
    "UserTenant",
    # Knowledge Base
    "Knowledgebase",
    "Document",
    "File",
    "File2Document",
    "Task",
    # Dialog
    "Dialog",
    "Conversation",
    # LLM
    "LLMFactories",
    "LLM",
    "TenantLLM",
    # Connector
    "Connector",
    "Connector2Kb",
    "SyncLogs",
    # Canvas
    "UserCanvas",
    "CanvasTemplate",
    # Evaluation
    "EvaluationDataset",
    "EvaluationCase",
    "EvaluationRun",
    "EvaluationResult",
    # System
    "SystemSettings",
    "APIToken",
    "API4Conversation",
    "MCP",
    "Search",
    "PipelineOperationLog",
]
