from __future__ import annotations

import time
from contextlib import ContextDecorator


class performance_timing(ContextDecorator):
    def __init__(self, label=None):
        self.label = label

    def __enter__(self):
        self.start_time = time.monotonic()
        return self

    def __exit__(self, *exc):
        time.monotonic() - self.start_time
        return False
