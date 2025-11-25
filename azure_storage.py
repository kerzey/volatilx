import json
import logging
import os
import re
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional

from azure.core.exceptions import ResourceExistsError  # type: ignore[import-untyped]
from azure.identity import DefaultAzureCredential  # type: ignore[import-untyped]
from azure.storage.blob import BlobServiceClient, ContentSettings  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class AzureStorageUnavailable(RuntimeError):
    """Raised when Azure storage configuration is missing."""


@lru_cache(maxsize=1)
def _get_blob_service_client():
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if connection_string:
        logger.debug("Initialising Azure Blob client via connection string")
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    else:
        account_url = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
        account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "volatilxairesults")
        if not account_url:
            if not account_name:
                raise AzureStorageUnavailable(
                    "Set AZURE_STORAGE_ACCOUNT_NAME or AZURE_STORAGE_ACCOUNT_URL when using managed identity."
                )
            account_url = f"https://{account_name}.blob.core.windows.net"

        try:
            credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
        except Exception as exc:  # noqa: BLE001 - include context for configuration issues
            raise AzureStorageUnavailable(f"Failed to initialise Azure credential: {exc}") from exc

        logger.debug("Initialising Azure Blob client via managed identity credential (account_url=%s)", account_url)
        blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)

    return blob_service_client


def _resolve_base_container() -> str:
    base_container = os.getenv("AZURE_STORAGE_CONTAINER", "ai-reports")
    sanitized_base = re.sub(r"[^a-z0-9-]", "-", base_container.lower())
    if not sanitized_base:
        sanitized_base = "ai-reports"
    return sanitized_base


def _build_container_name(target_date: datetime) -> str:
    sanitized_base = _resolve_base_container()
    container_name = f"{sanitized_base}-{target_date.strftime('%Y-%m-%d')}".strip("-")
    container_name = container_name[:63]
    if len(container_name) < 3:
        container_name = f"{container_name}{'a' * (3 - len(container_name))}"
    return container_name


@lru_cache(maxsize=64)
def _get_container_client(container_name: str):
    blob_service_client = _get_blob_service_client()

    container_client = blob_service_client.get_container_client(container_name)
    try:
        container_client.create_container()
    except ResourceExistsError:
        pass

    return container_client


def store_ai_report(symbol: str, user_id: Any, payload: Dict[str, Any]) -> bool:
    """Persist AI analysis output as JSON in Azure Blob Storage."""

    try:
        container_name = _build_container_name(datetime.utcnow())
        container_client = _get_container_client(container_name)
    except AzureStorageUnavailable as exc:
        logger.warning("Azure storage disabled: %s", exc)
        return False
    except Exception as exc:  # noqa: BLE001 - ensure storage failures never crash job
        logger.exception("Failed to initialise Azure storage client")
        return False

    safe_symbol = (symbol or "UNKNOWN").upper().replace("/", "-").replace(" ", "_")
    user_part = str(user_id)
    blob_name = f"{safe_symbol}_{user_part}.json"

    enriched_payload = {
        **payload,
        "symbol": safe_symbol,
        "user_id": user_part,
        "stored_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    try:
        json_bytes = json.dumps(enriched_payload, separators=(",", ":"), default=_json_fallback).encode("utf-8")
        content_settings = ContentSettings(content_type="application/json")
        container_client.upload_blob(
            name=blob_name,
            data=json_bytes,
            overwrite=True,
            content_settings=content_settings,
        )
        logger.info("Stored AI report %s in Azure container %s", blob_name, container_client.container_name)
        return True
    except Exception as exc:  # noqa: BLE001 - log error but do not interrupt main flow
        logger.exception("Failed to upload AI report to Azure storage: %s", exc)
        return False


def _json_fallback(value):
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def fetch_reports_for_date(
    target_date: datetime,
    *,
    symbol: Optional[str] = None,
    max_reports: int = 120,
) -> List[Dict[str, Any]]:
    """Retrieve stored AI reports for a given UTC date from Azure storage."""

    try:
        container_name = _build_container_name(target_date)
        blob_service_client = _get_blob_service_client()
        container_client = blob_service_client.get_container_client(container_name)
        if not container_client.exists():
            logger.info("Azure container %s does not exist yet for date %s", container_name, target_date.date())
            return []
    except AzureStorageUnavailable as exc:
        logger.warning("Azure storage unavailable: %s", exc)
        return []
    except Exception as exc:  # noqa: BLE001 - never propagate storage errors
        logger.exception("Failed to connect to Azure storage for Report Center fetch")
        return []

    prefix = None
    if symbol:
        safe_symbol = symbol.upper().replace("/", "-").replace(" ", "_")
        prefix = f"{safe_symbol}_"

    results: List[Dict[str, Any]] = []
    try:
        blob_iter = container_client.list_blobs(name_starts_with=prefix)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to list blobs in container %s", container_name)
        return []

    for blob in blob_iter:
        if max_reports and len(results) >= max_reports:
            break
        if not str(blob.name).lower().endswith(".json"):
            continue
        try:
            downloader = container_client.download_blob(blob.name)
            payload = downloader.readall()
            data = json.loads(payload)
            data.setdefault("_blob_name", blob.name)
            data.setdefault("_blob_last_modified", blob.last_modified.isoformat() if blob.last_modified else None)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping blob %s due to read error: %s", blob.name, exc)
            continue
        results.append(data)

    return results
