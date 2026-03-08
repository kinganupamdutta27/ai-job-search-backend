"""Celery application configuration.

Uses Redis as both broker and result backend.
Start the worker with:
    celery -A celery_app worker --loglevel=info --beat
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

app = Celery(
    "jobsearch",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    beat_schedule={},
)

app.autodiscover_tasks(["tasks"])
