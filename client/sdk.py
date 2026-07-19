"""Python client SDK for the deployed Document Analyst (Part 3).

TODO: Implement `DocumentAnalystClient` and `AnalystClientError` per Task 3.1:
  - __init__(endpoint_name, host=None, token=None, timeout=120.0, max_retries=3):
    read DATABRICKS_HOST/DATABRICKS_TOKEN from env when not provided.
  - ask(question) -> str
  - ask_streaming(question) -> Iterator[str]   (yield chunks as they arrive)
  - health_check() -> bool                      (True only when endpoint READY)
  - exponential backoff on 429/503, TimeoutError with elapsed time, and wrap HTTP
    errors in AnalystClientError(status_code, message, request_id).
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator

import requests


class AnalystClientError(Exception):
    """Custom error with HTTP status code, message, and optional request ID."""
    def __init__(self, message: str, status_code: int = None, request_id: str = None):
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id


class DocumentAnalystClient:
    """Reusable client for the Document Analyst serving endpoint."""
    def __init__(
        self,
        endpoint_name: str,
        host: str | None = None,
        token: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self.endpoint_name = endpoint_name
        self.host = host or os.environ["DATABRICKS_HOST"]
        self.token = token or os.environ["DATABRICKS_TOKEN"]
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = f"{self.host}/serving-endpoints/{self.endpoint_name}"

    def health_check(self) -> bool:
        """Return True if the endpoint is in READY state."""
        try:
            from databricks.sdk import WorkspaceClient

            w = WorkspaceClient()
            ep = w.serving_endpoints.get(self.endpoint_name)
            ready = ep.state.ready
            if hasattr(ready, "value"):
                ready = ready.value
            return ready == "READY"
        except Exception:
            return False

    def ask(self, question: str) -> str:
        """Send a question and return the synthesised answer."""
        payload = {"messages": [{"role": "user", "content": question}]}
        resp = self._request_with_retry(payload)
        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            state = data[0]
            messages = state.get("messages", [])
            if messages:
                return messages[-1].get("content", "")
        return str(data)

    def ask_streaming(self, question: str) -> Iterator[str]:
        """
        Yield text chunks from the endpoint.
        The v1 endpoint may not provide token‑by‑token streaming;
        in that case the full answer is yielded as a single chunk.
        """
        payload = {"messages": [{"role": "user", "content": question}]}
        resp = self._request_with_retry(payload, stream=True)
        content_type = resp.headers.get("content-type", "")

        if "text/event-stream" in content_type:
            for line in resp.iter_lines(decode_unicode=True):
                if line and line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        if "choices" in chunk:
                            token = chunk["choices"][0].get("delta", {}).get("content", "")
                            if token:
                                yield token
                    except json.JSONDecodeError:
                        pass
        else:
            answer = ""
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                msgs = data[0].get("messages", [])
                if msgs:
                    answer = msgs[-1].get("content", "")
            yield answer

    def _request_with_retry(self, payload: dict, stream: bool = False):
        """Send POST to /invocations with retry and exponential backoff."""
        url = f"{self.base_url}/invocations"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        backoff = 1.0
        last_exc = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                    stream=stream,
                )
                if resp.status_code in (429, 503):
                    if attempt < self.max_retries:
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                resp.raise_for_status()
                return resp
            except requests.Timeout as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    raise TimeoutError(
                        f"Request timed out after {self.timeout}s "
                        f"and {attempt + 1} attempts."
                    ) from exc
            except requests.HTTPError as exc:
                raise AnalystClientError(
                    message=exc.response.text,
                    status_code=exc.response.status_code,
                    request_id=exc.response.headers.get("x-request-id"),
                ) from exc

        if last_exc:
            raise TimeoutError("Request timed out after all retries.") from last_exc