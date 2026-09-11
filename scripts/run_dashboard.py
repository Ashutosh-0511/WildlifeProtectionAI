from __future__ import annotations

import asyncio
import sys

import uvicorn


def configure_windows_event_loop() -> None:
    """Avoid noisy Proactor socket-reset tracebacks during browser video range requests."""
    if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


if __name__ == "__main__":
    configure_windows_event_loop()
    uvicorn.run("apps.api.main:app", host="127.0.0.1", port=8000, reload=False)
