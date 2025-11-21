import json
import logging
import os
import re
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict

from azure.core.exceptions import ResourceExistsError  # type: ignore[import-untyped]
from azure.identity import DefaultAzureCredential  # type: ignore[import-untyped]
from azure.storage.blob import BlobServiceClient, ContentSettings  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class AzureStorageUnavailable(RuntimeError):
    """Raised when Azure storage configuration is missing."""


@lru_cache(maxsize=64)
def _get_container_client(container_name: str):
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

    container_client = blob_service_client.get_container_client(container_name)
    try:
        container_client.create_container()
    except ResourceExistsError:
        pass

    return container_client


def store_ai_report(symbol: str, user_id: Any, payload: Dict[str, Any]) -> bool:
    """Persist AI analysis output as JSON in Azure Blob Storage."""

    try:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        base_container = os.getenv("AZURE_STORAGE_CONTAINER", "ai-reports")
        sanitized_base = re.sub(r"[^a-z0-9-]", "-", base_container.lower())
        if not sanitized_base:
            sanitized_base = "ai-reports"
        container_name = f"{sanitized_base}-{today}".strip("-")
        container_name = container_name[:63]
        if len(container_name) < 3:
            container_name = f"{container_name}{'a' * (3 - len(container_name))}"  # pad to minimum length
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
