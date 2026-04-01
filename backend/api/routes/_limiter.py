# Rate limiter shared across all route modules.
# Created before any router imports to avoid circular dependency issues.
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.config import settings

limiter = Limiter(key_func=get_remote_address, enabled=bool(settings.api_key))
