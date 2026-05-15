"""
MinIO (S3-compatible) object storage client for RAGEve.

Provides async interface for file operations used by the ingestion pipeline.
Uses boto3 with proper signature version for MinIO compatibility.
"""

from __future__ import annotations

import logging
from typing import Any

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from backend.config_loader import get_settings
from backend.utils.log_sanitizer import sanitize_key

_log = logging.getLogger(__name__)


class MinIOClient:
    """S3-compatible client for MinIO object storage."""

    def __init__(self) -> None:
        settings = get_settings()
        self.endpoint = f"http://{settings.minio.host}"
        self.access_key = settings.minio.user
        self.secret_key = settings.minio.password
        self.secure = settings.minio.secure
        self.bucket = settings.minio.bucket
        self.prefix = settings.minio.prefix_path or ""
        self._bucket_checked = False

        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(
                signature_version="s3v4",
                s3={
                    "payload_signing_enabled": False,
                    "payload_checksum_algorithm": None,
                },
            ),
            use_ssl=self.secure,
        )

    def _ensure_bucket(self) -> None:
        """Create bucket if it doesn't exist - fails silently on connection errors."""
        if self._bucket_checked:
            return  # Skip repeated checks
        self._bucket_checked = True

        try:
            self.client.head_bucket(Bucket=self.bucket)
            _log.debug("MinIO bucket '%s' exists", sanitize_key(self.bucket))
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "404":
                try:
                    self.client.create_bucket(Bucket=self.bucket)
                    _log.info("Created MinIO bucket '%s'", sanitize_key(self.bucket))
                except ClientError as create_err:
                    _log.warning("Could not create MinIO bucket '%s': %s", sanitize_key(self.bucket), create_err)
            elif error_code == "403":
                _log.warning(
                    "Access denied for MinIO bucket '%s' - check credentials", sanitize_key(self.bucket)
                )
            else:
                _log.warning("Could not verify MinIO bucket '%s': %s", sanitize_key(self.bucket), e)
        except Exception as e:
            # Catch any other exceptions (connection errors, timeouts, etc.)
            _log.warning("Could not connect to MinIO to verify bucket '%s': %s", sanitize_key(self.bucket), e)

    def _get_key(self, path: str) -> str:
        """Build full object key with prefix."""
        return f"{self.prefix}{path}".lstrip("/")

    async def upload_file(
        self, key: str, data: bytes, content_type: str | None = None
    ) -> str:
        """
        Upload bytes to MinIO.

        Args:
            key: Object key (path within bucket)
            data: File content as bytes
            content_type: Optional MIME type

        Returns:
            Full S3 URL to the uploaded object
        """
        self._ensure_bucket()
        full_key = self._get_key(key)
        extra_args: dict[str, Any] = {}
        if content_type:
            extra_args["ContentType"] = content_type

        try:
            # Use bytes directly - BytesIO can cause signature mismatch with MinIO
            self.client.put_object(
                Bucket=self.bucket,
                Key=full_key,
                Body=data,
                ContentLength=len(data),
                **extra_args,
            )
            url = f"{self.endpoint}/{self.bucket}/{full_key}"
            _log.debug("Uploaded to MinIO: %s", sanitize_key(full_key))
            return url
        except ClientError as e:
            _log.error("Failed to upload %s: %s", sanitize_key(full_key), e)
            raise

    async def download_file(self, key: str) -> bytes:
        """
        Download file from MinIO.

        Args:
            key: Object key (path within bucket)

        Returns:
            File content as bytes
        """
        self._ensure_bucket()
        full_key = self._get_key(key)
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=full_key)
            data = response["Body"].read()
            _log.debug("Downloaded from MinIO: %s", sanitize_key(full_key))
            return data
        except ClientError as e:
            _log.error("Failed to download %s: %s", sanitize_key(full_key), e)
            raise

    async def delete_file(self, key: str) -> None:
        """
        Delete file from MinIO.

        Args:
            key: Object key (path within bucket)
        """
        full_key = self._get_key(key)
        try:
            self.client.delete_object(Bucket=self.bucket, Key=full_key)
            _log.debug("Deleted from MinIO: %s", full_key)
        except ClientError as e:
            _log.error("Failed to delete %s: %s", full_key, e)
            raise

    async def list_files(self, prefix: str = "") -> list[str]:
        """
        List files under a prefix.

        Args:
            prefix: Path prefix to list (e.g., "uploads/")

        Returns:
            List of object keys
        """
        self._ensure_bucket()
        full_prefix = self._get_key(prefix)
        keys: list[str] = []
        try:
            paginator = self.client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])
            return keys
        except ClientError as e:
            _log.error("Failed to list %s: %s", full_prefix, e)
            raise

    def get_upload_path(self, dataset_id: str, filename: str) -> str:
        """Generate MinIO key for an uploaded file."""
        return f"uploads/{dataset_id}/{filename}"

    def get_chunk_path(
        self, dataset_id: str, source_file: str, chunk_index: int
    ) -> str:
        """Generate MinIO key for a chunk file."""
        stem = source_file.rsplit(".", 1)[0] if "." in source_file else source_file
        return f"chunks/{dataset_id}/{stem}.chunk-{chunk_index:04d}.txt"

    def get_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        """
        Generate a presigned URL for temporary file access.

        Args:
            key: Object key
            expires_in: Expiration time in seconds (default 1 hour)

        Returns:
            Presigned URL
        """
        full_key = self._get_key(key)
        url = self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": full_key},
            ExpiresIn=expires_in,
        )
        return url


# Singleton instance
_minio_client: MinIOClient | None = None


def get_minio_client() -> MinIOClient:
    """Get or create MinIO client singleton."""
    global _minio_client
    if _minio_client is None:
        _minio_client = MinIOClient()
    return _minio_client
