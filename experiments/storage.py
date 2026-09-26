"""Small local persistence primitives shared by training and evaluation."""
from contextlib import contextmanager
import errno
import fcntl
import json
import os
from pathlib import Path
import socket
import tempfile


def json_bytes(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    except OSError as exc:
        if exc.errno not in (errno.EINVAL, errno.ENOTSUP):
            raise
    finally:
        os.close(descriptor)


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    atomic_bytes(path, json_bytes(value))


def publish_json(path, value):
    """Complete an interrupted export, but never replace different evidence."""
    publish_bytes(path, json_bytes(value))


def publish_bytes(path, payload):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Existing output differs from committed progress: {path}")
    else:
        atomic_bytes(path, payload)


@contextmanager
def run_lock(directory, name=".run.lock"):
    with (Path(directory) / name).open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This output already has an active training or experiment process") from exc
        handle.seek(0)
        handle.truncate()
        json.dump({"pid": os.getpid(), "host": socket.gethostname()}, handle)
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
