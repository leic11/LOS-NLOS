from __future__ import annotations

import sys
from pathlib import Path


class Tee:
    """Write stdout/stderr to multiple streams."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            try:
                stream.write(data)
                stream.flush()
            except Exception:
                pass

    def flush(self) -> None:
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass


class RedirectStdStreams:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_file = None
        self.old_out = None
        self.old_err = None

    def __enter__(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_path, "w", encoding="utf-8")
        self.old_out, self.old_err = sys.stdout, sys.stderr
        sys.stdout = Tee(self.old_out, self.log_file)
        sys.stderr = Tee(self.old_err, self.log_file)
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.stdout = self.old_out
        sys.stderr = self.old_err
        if self.log_file is not None:
            self.log_file.close()
        print(f"[RUN] Log saved to: {self.log_path}")
