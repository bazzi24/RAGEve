"""
Base model and custom field types for Peewee ORM.

Provides:
- BaseModel: all models inherit from this
- JSONTextField: stores JSON as LONGTEXT
- ListField: stores lists as JSON
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import peewee

_log = logging.getLogger(__name__)


class JSONTextField(peewee.TextField):
    """JSON field stored as LONGTEXT in MySQL."""

    field_type = "LONGTEXT"

    def db_value(self, value: Any) -> str | None:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False)

    def python_value(self, value: str | None) -> Any:
        if not value:
            return {}
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            # Log warning with field context for debugging malformed JSON
            _log.warning(
                "JSONTextField: Failed to decode JSON value (truncated: %s)",
                value[:100] if isinstance(value, str) else type(value),
            )
            # Return empty dict as safe fallback for non-critical fields
            return {}


class ListField(JSONTextField):
    """Array field using JSON. Ensures empty list instead of empty dict."""

    def db_value(self, value: list | None) -> str:
        if value is None:
            return "[]"
        return json.dumps(value, ensure_ascii=False)

    def python_value(self, value: str | None) -> list:
        if not value:
            return []
        try:
            result = json.loads(value)
            return result if isinstance(result, list) else []
        except (json.JSONDecodeError, TypeError):
            return []


class BaseModel(peewee.Model):
    """Base class for all database models.

    Includes automatic timestamp tracking:
    - create_time: Unix timestamp (bigint) when record created
    - create_date: datetime when record created
    - update_time: Unix timestamp (bigint) when record last updated
    - update_date: datetime when record last updated
    """

    create_time = peewee.BigIntegerField(null=True, index=True)
    create_date = peewee.DateTimeField(null=True, index=True)
    update_time = peewee.BigIntegerField(null=True, index=True)
    update_date = peewee.DateTimeField(null=True, index=True)

    class Meta:
        database = None  # Will be set by the database singleton

    def to_dict(self) -> dict[str, Any]:
        """Convert model instance to dictionary."""
        data = self.__dict__.get("__data__", {}).copy()
        # Convert datetime fields to ISO format strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data

    @classmethod
    def to_dict_many(cls, instances: list[BaseModel]) -> list[dict[str, Any]]:
        """Convert multiple model instances to list of dictionaries."""
        return [inst.to_dict() for inst in instances]
