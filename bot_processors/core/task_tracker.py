"""Shared fire-and-forget background task tracker.

asyncio only holds a *weak* reference to Task objects — a bare
asyncio.create_task(...) with no other reference can be garbage-collected
mid-flight, silently dropping its work. Tasks tracked here are held in a
module-level set until they finish (removing themselves via
add_done_callback), so they always run to completion regardless of what
else holds a reference.
"""

import asyncio

_background_tasks: set[asyncio.Task] = set()


def track_task(coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task
