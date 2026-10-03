"""Atomic artifact writes so interrupted runs do not destroy a usable cache."""
import json
import os
import tempfile
from pathlib import Path


def _atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_csv(df, path):
    _atomic_write(path, lambda name: df.to_csv(name, index=False, encoding="utf-8-sig"))


def write_json(data, path):
    def writer(name):
        with open(name, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
    _atomic_write(path, writer)
