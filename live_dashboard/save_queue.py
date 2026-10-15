"""Bounded producer/writer handoff. Only the writer touches publication storage."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from queue import Empty, Full, Queue
from threading import Event, Thread
from typing import TypeVar

T = TypeVar("T")
_END = object()


def stream_to_writer[T](
    items: Iterable[T], write: Callable[[T], None], *, capacity: int = 2
) -> int:
    """Drain on producer failure; propagate writer failure without a blocked producer.

    Ownership of an item transfers at enqueue: producers must never mutate it again.
    At most capacity queued items + one being written + one being generated are live.
    Storage must provide bounded calls (Modal's container timeout is the final bound).
    This function joins its writer before returning or raising; callers may only
    refresh/read the shared Volume after it returns.
    """
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        raise ValueError("queue capacity must be a positive integer")
    queue: Queue = Queue(maxsize=capacity)
    failed = Event()
    errors: list[BaseException] = []
    completed = 0

    def consume():
        nonlocal completed
        try:
            while True:
                item = queue.get()
                if item is _END:
                    return
                write(item)
                completed += 1
                del item
        except BaseException as error:
            errors.append(error)
            failed.set()

    thread = Thread(target=consume, name="publication-writer", daemon=True)
    thread.start()

    def put(item):
        while not failed.is_set():
            try:
                queue.put(item, timeout=0.05)
                return
            except Full:
                pass
        raise RuntimeError("publication writer failed") from errors[0]

    try:
        for item in items:
            put(item)
            del item
    finally:
        # Even when generation raises, publish already handed-off batches.
        if not failed.is_set():
            try:
                put(_END)
            except RuntimeError:
                pass
        thread.join()
        # Drop unwritten arrays after failure, not just their identifiers.
        while True:
            try:
                queue.get_nowait()
            except Empty:
                break
        if errors:
            raise RuntimeError("publication writer failed") from errors[0]
    return completed
