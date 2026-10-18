"""Single-submission native tools, same-task recovery and service-specific terminal checks."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlsplit

from acedatacloud import AceDataCloud, AsyncAceDataCloud
from llama_index.core.tools.tool_spec.base import BaseToolSpec

API_BASE = "https://api.acedata.cloud"
_FAILED = {"failed", "failure", "error", "cancelled", "canceled", "rejected"}


def _required(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must contain text")
    return value.strip()


def _url(value: str) -> str:
    value = _required(value, "image URL")
    parsed = urlsplit(value)
    if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError(
            "Use a public HTTP(S) image URL without embedded credentials; local files/data URIs are not accepted here"
        )
    return value


def _submitted(task_id: str) -> dict[str, Any]:
    if not task_id:
        raise RuntimeError("No task ID returned. Outcome unknown: inspect task history, do not automatically resubmit.")
    return {
        "task_id": task_id,
        "status": "pending",
        "success": None,
        "media_urls": [],
        "next_tool": "gpt_image_get_task",
        "polling_interval_seconds": 15,
        "message": "Save this ID. Query the same task after 15 seconds; never generate again to check progress.",
    }


def _task_id(raw: dict[str, Any]) -> str:
    value = raw.get("task_id") or raw.get("id")
    data = raw.get("data")
    if not value and isinstance(data, dict):
        value = data.get("task_id")
    return str(value or "")


def _result(task_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    if not raw:
        return {
            "task_id": task_id,
            "status": "not_found",
            "success": None,
            "media_urls": [],
            "trace_id": None,
            "polling_interval_seconds": None,
            "message": "No task matched this ID. Verify ID and application access; do not submit a new task.",
        }
    returned_id = raw.get("id") or raw.get("task_id")
    if returned_id and str(returned_id) != task_id:
        raise ValueError("Task response ID differs from the requested ID")
    response = raw.get("response", raw)
    if response is None:
        response = {}
    if not isinstance(response, dict):
        raise ValueError("Task response must be an object; inspect the same task, do not resubmit")
    data = response.get("data", [])
    items = data if isinstance(data, list) else [data]
    words = {
        str(node.get(key, "")).lower()
        for node in [raw, response, *items]
        if isinstance(node, dict)
        for key in ("state", "status")
    }
    if raw.get("finished_at") is None:
        status, success = "pending", None
    elif words & _FAILED or response.get("success") is False or response.get("error"):
        status, success = "failed", False
    else:
        # The task store stamps finished_at only when the image operation settles.
        status, success = "succeeded", True
    urls: list[str] = []
    if status == "succeeded":
        for item in items:
            if isinstance(item, dict) and isinstance(item.get("url"), str):
                url = item["url"]
                parsed = urlsplit(url)
                if (
                    parsed.scheme in {"https", "http"}
                    and parsed.hostname
                    and not parsed.username
                    and not parsed.password
                ):
                    if url not in urls:
                        urls.append(url)
        if not urls:
            raise ValueError("Task reports success without final media URLs; inspect the same task, do not resubmit")
    output = {
        "task_id": task_id,
        "status": status,
        "success": success,
        "media_urls": urls,
        "trace_id": raw.get("trace_id") or response.get("trace_id"),
        "polling_interval_seconds": 15 if status == "pending" else None,
    }
    if status == "failed":
        output["error"] = response.get("error") or raw.get("error") or "Task failed"
    return output


class GPTImageToolSpec(BaseToolSpec):
    """GPT Image native tools provided by Ace Data Cloud; explicit model and no generation retries.

    Construction and to_tool_list send no request. Generation may consume credits.
    """

    spec_functions = [
        ("gpt_image_generate_image", "agpt_image_generate_image"),
        ("gpt_image_edit_image", "agpt_image_edit_image"),
        ("gpt_image_get_task", "agpt_image_get_task"),
    ]

    def __init__(self, api_token: str | None = None) -> None:
        token = api_token if api_token is not None else os.environ.get("ACEDATACLOUD_API_TOKEN", "")
        token = _required(token, "ACEDATACLOUD_API_TOKEN")
        if token.lower().startswith("bearer ") or token.startswith("platform-"):
            raise ValueError("Use a raw application API token, without Bearer; not an account-management token")
        self._client = AceDataCloud(api_token=token, base_url=API_BASE, max_retries=0)
        self._async_client = AsyncAceDataCloud(api_token=token, base_url=API_BASE, max_retries=0)

    def gpt_image_generate_image(
        self, prompt: str, model: str, size: str, quality: str = "auto", count: int = 1
    ) -> dict[str, Any]:
        """Submit GPT Image image generation once. Consumes credits; query the same returned task ID."""
        prompt, model = _required(prompt, "prompt"), _required(model, "model")
        if not model.startswith("gpt-image-"):
            raise ValueError("This brand tool accepts explicit GPT Image models only")
        if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 10:
            raise ValueError("count must be an integer from 1 through 10")
        if quality not in {"auto", "low", "medium", "high"}:
            raise ValueError("Unsupported GPT Image quality")
        size = _required(size, "size")
        raw = self._client.openai.images.generate(
            prompt=prompt, model=model, size=size, quality=quality, n=count, response_format="url", async_=True
        )
        return _submitted(_task_id(raw))

    async def agpt_image_generate_image(
        self, prompt: str, model: str, size: str, quality: str = "auto", count: int = 1
    ) -> dict[str, Any]:
        """Submit GPT Image image generation once. Consumes credits; query the same returned task ID."""
        prompt, model = _required(prompt, "prompt"), _required(model, "model")
        if not model.startswith("gpt-image-"):
            raise ValueError("This brand tool accepts explicit GPT Image models only")
        if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 10:
            raise ValueError("count must be an integer from 1 through 10")
        if quality not in {"auto", "low", "medium", "high"}:
            raise ValueError("Unsupported GPT Image quality")
        size = _required(size, "size")
        raw = await self._async_client.openai.images.generate(
            prompt=prompt, model=model, size=size, quality=quality, n=count, response_format="url", async_=True
        )
        return _submitted(_task_id(raw))

    def gpt_image_edit_image(
        self, image_urls: list[str], prompt: str, model: str, size: str, quality: str = "auto", count: int = 1
    ) -> dict[str, Any]:
        """Submit GPT Image image editing once. Consumes credits; query the same returned task ID."""
        prompt, model = _required(prompt, "prompt"), _required(model, "model")
        if not model.startswith("gpt-image-"):
            raise ValueError("This brand tool accepts explicit GPT Image models only")
        if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 10:
            raise ValueError("count must be an integer from 1 through 10")
        if quality not in {"auto", "low", "medium", "high"}:
            raise ValueError("Unsupported GPT Image quality")
        size = _required(size, "size")
        if not isinstance(image_urls, list) or not image_urls:
            raise ValueError("At least one input image URL is required for editing")
        images = [_url(url) for url in image_urls]
        raw = self._client.openai.images.edit(
            image=images,
            prompt=prompt,
            model=model,
            size=size,
            quality=quality,
            n=count,
            response_format="url",
            async_=True,
        )
        return _submitted(_task_id(raw))

    async def agpt_image_edit_image(
        self, image_urls: list[str], prompt: str, model: str, size: str, quality: str = "auto", count: int = 1
    ) -> dict[str, Any]:
        """Submit GPT Image image editing once. Consumes credits; query the same returned task ID."""
        prompt, model = _required(prompt, "prompt"), _required(model, "model")
        if not model.startswith("gpt-image-"):
            raise ValueError("This brand tool accepts explicit GPT Image models only")
        if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 10:
            raise ValueError("count must be an integer from 1 through 10")
        if quality not in {"auto", "low", "medium", "high"}:
            raise ValueError("Unsupported GPT Image quality")
        size = _required(size, "size")
        if not isinstance(image_urls, list) or not image_urls:
            raise ValueError("At least one input image URL is required for editing")
        images = [_url(url) for url in image_urls]
        raw = await self._async_client.openai.images.edit(
            image=images,
            prompt=prompt,
            model=model,
            size=size,
            quality=quality,
            n=count,
            response_format="url",
            async_=True,
        )
        return _submitted(_task_id(raw))

    def gpt_image_get_task(self, task_id: str) -> dict[str, Any]:
        """Retrieve an existing GPT Image task once; never generates. Save/reuse the same task_id.

        Pending preview URLs are hidden. Wait at least 15 seconds before another pending query.
        A new tool instance can recover a prior task ID with the same application's authorized token.
        """
        task_id = _required(task_id, "task_id")
        return _result(task_id, self._client.openai.tasks.retrieve(id=task_id))

    async def agpt_image_get_task(self, task_id: str) -> dict[str, Any]:
        """Read the same existing GPT Image task asynchronously, without an internal polling loop."""
        task_id = _required(task_id, "task_id")
        return _result(task_id, await self._async_client.openai.tasks.retrieve(id=task_id))

    def close(self) -> None:
        """Close the synchronous client."""
        self._client.close()

    async def aclose(self) -> None:
        """Close both HTTP clients after sync or async usage."""
        self._client.close()
        await self._async_client.close()
